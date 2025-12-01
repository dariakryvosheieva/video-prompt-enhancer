import argparse, json, logging, random, time, hashlib, signal, threading, io
from pathlib import Path

import torch, numpy as np
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from openai import OpenAI
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import tqdm
from decord import cpu, VideoReader, bridge

from train_stage1 import INSTRUCTION_TEXT


LOG = logging.getLogger("stage2")
STATE_FILENAME = "stage2_state.pt"


def compute_prompts_hash(prompts):
    digest = hashlib.sha256()
    for text in prompts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if np is not None:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])


def load_training_state(path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def save_training_state_metadata(
    path,
    *,
    prompt_indices,
    prompt_cursor,
    global_step,
    dataset_hash,
):
    metadata = {
        "prompt_indices": prompt_indices,
        "prompt_cursor": prompt_cursor,
        "global_step": global_step,
        "dataset_hash": dataset_hash,
        "rng_state": capture_rng_state(),
    }
    torch.save(metadata, path)


def register_signal_handlers(
    stop_event,
    checkpoint_fn=None,
):
    timeout_signals = []
    for maybe in ("SIGUSR1", "SIGXCPU"):
        sig_obj = getattr(signal, maybe, None)
        if isinstance(sig_obj, int):
            timeout_signals.append(sig_obj)

    def _handler(signum, _frame):
        LOG.warning("Received signal %s; checkpointing and preparing to exit.", signum)
        stop_event.set()
        if checkpoint_fn is not None:
            checkpoint_fn(f"signal-{signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, *timeout_signals):
        signal.signal(sig, _handler)


def save_checkpoint(
    *,
    output_dir,
    tokenizer,
    ppo_trainer,
    prompt_indices,
    prompt_cursor,
    global_step,
    dataset_hash,
    reason,
):
    if not ppo_trainer.accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "Saving checkpoint (%s) at step %d (processed %d / %d prompts).",
        reason,
        global_step,
        prompt_cursor,
        len(prompt_indices),
    )
    ppo_trainer.save_pretrained(output_dir)
    try:
        if hasattr(ppo_trainer.model, "pretrained_model") and hasattr(
            ppo_trainer.model.pretrained_model, "save_pretrained"
        ):
            ppo_trainer.model.pretrained_model.save_pretrained(output_dir)
    except Exception:
        LOG.warning("Failed to save LoRA adapter")
    try:
        v_head_path = output_dir / "v_head.pt"
        state_dict = ppo_trainer.model.v_head.state_dict()
        torch.save(state_dict, v_head_path)
    except Exception:
        LOG.warning("Failed to save value head")
    tokenizer.save_pretrained(output_dir)
    save_training_state_metadata(
        output_dir / STATE_FILENAME,
        prompt_indices=prompt_indices,
        prompt_cursor=prompt_cursor,
        global_step=global_step,
        dataset_hash=dataset_hash,
    )


def cleanup_state_file(output_dir):
    state_file = output_dir / STATE_FILENAME
    if state_file.exists():
        state_file.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 RL (PPO) training with Sora feedback."
    )
    parser.add_argument("--data_dir", type=str, default="data_stage2")
    parser.add_argument(
        "--adapter_dir", type=str, default="out/qwen2.5-14b-prompt-enhancer-lora"
    )
    parser.add_argument(
        "--output_dir", type=str, default="out/qwen2.5-14b-prompt-enhancer-lora-stage2"
    )
    parser.add_argument("--video_cache_dir", type=str, default="out/video_cache")
    parser.add_argument("--sample_log", type=str, default="out/stage2_traces.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--retry_limit", type=int, default=3)
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--target_kl", type=float, default=0.1)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--clip_range_value", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--sora_poll_interval", type=float, default=10.0)
    parser.add_argument("--sora_timeout", type=float, default=1800.0)
    parser.add_argument("--vr_timeout", type=float, default=1800.0)
    return parser.parse_args()


def discover_json_shards(data_dir):
    path = Path(data_dir)
    shards = sorted(path.glob("*.json"))
    return shards


def load_simple_prompts(data_dir):
    prompts = []
    for shard in discover_json_shards(data_dir):
        with shard.open("r", encoding="utf-8") as f:
            data = json.load(f)
        shard_prompts = [p.strip() for p in data if isinstance(p, str) and p.strip()]
        prompts.extend(shard_prompts)
    unique_prompts = [p for p in dict.fromkeys(prompts)]
    LOG.info("Loaded %d simple prompts from %s", len(unique_prompts), data_dir)
    return unique_prompts


def format_query(simple_prompt):
    return f"{INSTRUCTION_TEXT}\n\nInput:\n{simple_prompt.strip()}\n\nOutput:\n"


class SoraClient:
    def __init__(
        self,
        cache_dir,
        poll_interval=5.0,
        timeout=600.0,
        stop_event=None,
    ):
        self.client = OpenAI()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.stop_event = stop_event or threading.Event()

    @staticmethod
    def _cache_key(prompt, seed):
        payload = f"{prompt}|{seed}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def generate_video(self, prompt, seed):
        cache_path = self.cache_dir / f"{self._cache_key(prompt, seed)}.mp4"
        if cache_path.exists():
            return cache_path

        if self.stop_event.is_set():
            raise RuntimeError("Termination requested before submitting Sora job.")

        job = self.client.videos.create(prompt=prompt)
        job_id = job.id

        deadline = time.monotonic() + self.timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Sora job {job_id} timed out.")
            if self.stop_event.is_set():
                raise RuntimeError(
                    f"Termination requested while waiting for Sora job {job_id}."
                )
            job = self.client.videos.retrieve(job_id)
            status = job.status
            if status == "completed":
                content = self.client.videos.download_content(job_id)
                with cache_path.open("wb") as out_f:
                    for chunk in content.iter_bytes():
                        out_f.write(chunk)
                return cache_path
            if status == "failed":
                detail = job.error.message if job.error else "unknown error"
                raise RuntimeError(f"Sora job {job_id} failed: {detail}")
            if self.stop_event.wait(self.poll_interval):
                raise RuntimeError(
                    f"Termination requested while waiting for Sora job {job_id}."
                )


class VisionRewardVideoScorer:
    def __init__(self, timeout_s=1800.0):
        self.timeout_s = timeout_s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_type = resolve_dtype()

        self.questions_path = "VisionReward_Video/VisionReward_video_qa_select.txt"
        with open(self.questions_path, "r") as f:
            self.questions = f.readlines()

        self.weight_path = "VisionReward_Video/weight.json"
        with open(self.weight_path, "r") as f:
            self.weight = json.load(f)

        self.model_path = "THUDM/VisionReward-Video"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            # padding_side="left"
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=self.torch_type, trust_remote_code=True
            )
            .eval()
            .to(self.device)
        )

    def load_video(self, video_data, strategy="chat"):
        bridge.set_bridge("torch")
        mp4_stream = video_data
        num_frames = 24
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

        frame_id_list = None
        total_frames = len(decord_vr)
        if strategy == "base":
            clip_end_sec = 60
            clip_start_sec = 0
            start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
            end_frame = (
                min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
                if clip_end_sec is not None
                else total_frames
            )
            frame_id_list = np.linspace(
                start_frame, end_frame - 1, num_frames, dtype=int
            )
        elif strategy == "chat":
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_id_list = []
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data

    def inference(self, video_path, query, temperature=0.1):
        video_data = open(video_path, "rb").read()
        strategy = "chat"
        video = self.load_video(video_data, strategy=strategy)

        history = []

        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy,
        )
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(self.device),
            "images": [[inputs["images"][0].to("cuda").to(self.torch_type)]],
        }

        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1]]

        return self.tokenizer.decode(outputs[0])

    def score_video(self, video_path, simple_prompt):
        queries = [
            question.replace("[[prompt]]", simple_prompt) for question in self.questions
        ]
        answers = []
        for query in tqdm(queries, "scoring video"):
            answer = self.inference(video_path, query)
            answers.append(answer)
        answers = np.array([1 if answer == "yes" else -1 for answer in answers])
        return np.mean(answers * self.weight).item()


class RewardLogger:
    def __init__(self, jsonl_path):
        self.path = jsonl_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload):
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")


def init_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def resolve_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_value_head_with_lora(
    base_model_name,
    adapter_dir,
    dtype,
    device_map,
    trainable,
):
    wrapper = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if not isinstance(wrapper.pretrained_model, PeftModel):
        wrapper.pretrained_model = PeftModel.from_pretrained(
            wrapper.pretrained_model,
            adapter_dir,
            is_trainable=trainable,
        )
    v_head_path = Path(adapter_dir) / "v_head.pt"
    if v_head_path.exists():
        v_head_state = torch.load(v_head_path, map_location="cpu")
        wrapper.v_head.load_state_dict(v_head_state, strict=True)
    if not trainable:
        wrapper.eval()
        for param in wrapper.parameters():
            param.requires_grad_(False)
    return wrapper


def instantiate_policy_and_ref(
    base_model_name,
    policy_adapter_dir,
    reference_adapter_dir,
    dtype,
    device_map,
):
    policy_model = _load_value_head_with_lora(
        base_model_name=base_model_name,
        adapter_dir=policy_adapter_dir,
        dtype=dtype,
        device_map=device_map,
        trainable=True,
    )
    policy_model.config.use_cache = False

    ref_model = _load_value_head_with_lora(
        base_model_name=base_model_name,
        adapter_dir=reference_adapter_dir,
        dtype=dtype,
        device_map=device_map,
        trainable=False,
    )
    return policy_model, ref_model


def prepare_ppo_trainer(
    policy_model,
    ref_model,
    tokenizer,
    args,
):
    batch_size = args.batch_size * args.samples_per_prompt
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        target_kl=args.target_kl,
        ppo_epochs=args.ppo_epochs,
        cliprange=args.clip_range,
        cliprange_value=args.clip_range_value,
        log_with=None,
        remove_unused_columns=False,
        seed=args.seed,
    )
    return PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )


def create_reward_stack(
    scorer,
    sora_client,
    response_text,
    simple_prompt,
    seed_value,
):
    if not (sora_client and scorer):
        raise RuntimeError("Sora client and scorer must be initialized.")
    video_path = sora_client.generate_video(response_text, seed_value)
    score = scorer.score_video(video_path, simple_prompt)
    return score, video_path


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    stop_event = threading.Event()
    register_signal_handlers(stop_event)
    set_seed(args.seed)

    prompts = load_simple_prompts(args.data_dir)
    dataset_hash = compute_prompts_hash(prompts)
    state_path = Path(args.output_dir) / STATE_FILENAME
    resume_state = load_training_state(state_path)

    prompt_indices = list(range(len(prompts)))
    prompt_cursor = 0
    global_step = 0
    policy_adapter_dir = args.adapter_dir

    if resume_state is not None:
        prompt_indices = resume_state["prompt_indices"]
        prompt_cursor = resume_state["prompt_cursor"]
        global_step = resume_state["global_step"]
        restore_rng_state(resume_state["rng_state"])
        policy_adapter_dir = args.output_dir
        LOG.info(
            "Resuming Stage-2 training from step %d (next prompt index %d).",
            global_step,
            prompt_cursor,
        )
    else:
        random.shuffle(prompt_indices)

    peft_cfg = PeftConfig.from_pretrained(policy_adapter_dir)
    base_model = peft_cfg.base_model_name_or_path
    tokenizer = init_tokenizer(base_model)
    dtype = resolve_dtype()

    policy_model, ref_model = instantiate_policy_and_ref(
        base_model_name=base_model,
        policy_adapter_dir=policy_adapter_dir,
        reference_adapter_dir=args.adapter_dir,
        dtype=dtype,
        device_map=args.device_map,
    )

    ppo_trainer = prepare_ppo_trainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=args,
    )

    reward_scorer = VisionRewardVideoScorer(
        timeout_s=args.vr_timeout,
    )
    sora_client = SoraClient(
        cache_dir=Path(args.video_cache_dir),
        poll_interval=args.sora_poll_interval,
        timeout=args.sora_timeout,
        stop_event=stop_event,
    )
    reward_logger = RewardLogger(Path(args.sample_log))

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    ppo_device = ppo_trainer.accelerator.device
    total_prompts = len(prompt_indices)
    output_dir = Path(args.output_dir)

    def checkpoint(reason):
        save_checkpoint(
            output_dir=output_dir,
            tokenizer=tokenizer,
            ppo_trainer=ppo_trainer,
            prompt_indices=prompt_indices,
            prompt_cursor=prompt_cursor,
            global_step=global_step,
            dataset_hash=dataset_hash,
            reason=reason,
        )

    register_signal_handlers(stop_event, checkpoint_fn=checkpoint)

    try:
        while global_step < args.total_steps:
            if stop_event.is_set():
                checkpoint("termination")
                break
            if prompt_cursor >= total_prompts:
                LOG.info(
                    "All prompts processed after %d PPO steps (target=%d).",
                    global_step,
                    args.total_steps,
                )
                break

            end = min(prompt_cursor + args.batch_size, total_prompts)
            batch_indices = prompt_indices[prompt_cursor:end]
            batch_prompts = [prompts[idx] for idx in batch_indices]

            query_tensors = []
            response_tensors = []
            rewards = []

            for simple_prompt in batch_prompts:
                query_text = format_query(simple_prompt)
                query_tensor = (
                    tokenizer(query_text, return_tensors="pt")
                    .input_ids[0]
                    .to(ppo_device)
                )
                query_length = query_tensor.shape[-1]
                sample_attempts = args.samples_per_prompt * args.retry_limit
                generated_samples = 0
                attempt_idx = 0
                while (
                    generated_samples < args.samples_per_prompt
                    and attempt_idx < sample_attempts
                ):
                    attempt_idx += 1
                    sample_seed = random.randint(0, 2**31 - 1)
                    set_seed(sample_seed)
                    try:
                        response = ppo_trainer.generate(
                            query_tensor,
                            **gen_kwargs,
                        )
                        completion = response[:, query_length:]
                        completion_text = tokenizer.decode(
                            completion[0], skip_special_tokens=True
                        ).strip()

                        score, video_path = create_reward_stack(
                            reward_scorer,
                            sora_client,
                            completion_text,
                            simple_prompt,
                            sample_seed,
                        )
                    except Exception as exc:
                        if stop_event.is_set():
                            raise
                        LOG.warning(
                            "Sample attempt %d/%d failed (prompt='%s'): %s",
                            attempt_idx,
                            sample_attempts,
                            simple_prompt,
                            exc,
                        )
                        continue

                    query_tensors.append(query_tensor.clone().detach())
                    response_tensors.append(completion[0].clone().detach())
                    rewards.append(
                        torch.tensor(score, dtype=torch.float32, device=ppo_device)
                    )
                    generated_samples += 1

                    if ppo_trainer.accelerator.is_main_process:
                        reward_logger.log(
                            {
                                "step": global_step,
                                "simple_prompt": simple_prompt,
                                "detailed_prompt": completion_text,
                                "reward": score,
                                "seed": sample_seed,
                                "video_path": str(video_path) if video_path else None,
                            }
                        )

                if generated_samples < args.samples_per_prompt:
                    LOG.warning(
                        "Prompt '%s' yielded only %d/%d samples after %d attempts; skipping batch.",
                        simple_prompt,
                        generated_samples,
                        args.samples_per_prompt,
                        sample_attempts,
                    )
                    rewards = []
                    break

            prompt_cursor = end

            if not rewards:
                LOG.warning("No rewards collected for this batch; continuing.")
                continue

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            global_step += 1

            if ppo_trainer.accelerator.is_main_process:
                reward_mean = stats.get("ppo/mean_scores")
                kl_value = stats.get("objective/kl")
                entropy_value = stats.get("objective/entropy")
                LOG.info(
                    "Step %d | reward=%s | kl=%s | entropy=%s",
                    global_step,
                    f"{float(reward_mean):.4f}" if reward_mean is not None else "n/a",
                    f"{float(kl_value):.4f}" if kl_value is not None else "n/a",
                    (
                        f"{float(entropy_value):.4f}"
                        if entropy_value is not None
                        else "n/a"
                    ),
                )

            checkpoint(f"step-{global_step:05d}")

    except Exception:
        if stop_event.is_set():
            checkpoint("termination")
        else:
            checkpoint("exception")
        raise

    if stop_event.is_set():
        checkpoint("termination")

    if ppo_trainer.accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        ppo_trainer.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        cleanup_state_file(output_dir)
        LOG.info("Training complete. Final adapter stored at %s", output_dir)
