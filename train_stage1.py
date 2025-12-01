import argparse, json, os
from glob import glob

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


INSTRUCTION_TEXT = (
    "Convert the following video generation prompt into a professional-grade prompt that will produce a high quality, aesthetic, and impressive video."
    "If the original prompt includes a style specification (such as 'anime', 'pixel', or 'cartoon'), keep it in the converted prompt."
    "Output only the converted prompt."
)


def discover_shards(data_dir):
    pattern = os.path.join(data_dir, "*.json")
    shards = sorted(glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No shards found at {pattern}")
    return shards


def load_pairs_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for key in sorted(data.keys()):
        entry = data[key]
        if "lay" in entry and "elaborate" in entry:
            pairs.append((entry["lay"], entry["elaborate"]))
    return pairs


def build_prompt(lay_prompt):
    return f"{INSTRUCTION_TEXT}\n\nInput:\n{lay_prompt.strip()}\n\nOutput:\n"


class PromptEnhancementDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_seq_length=1024,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eos_token = tokenizer.eos_token or ""

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        lay_prompt, elaborate_prompt = self.rows[idx]
        prompt_text = build_prompt(lay_prompt)
        response_text = elaborate_prompt.strip()

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length - 32,
        )["input_ids"]

        response_ids = self.tokenizer(
            response_text + self.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length - len(prompt_ids),
        )["input_ids"]

        input_ids = prompt_ids + response_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + response_ids

        if len(input_ids) > self.max_seq_length:
            overflow = len(input_ids) - self.max_seq_length
            input_ids = input_ids[overflow:]
            attention_mask = attention_mask[overflow:]
            labels = labels[overflow:]
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[: self.max_seq_length]
                attention_mask = attention_mask[: self.max_seq_length]
                labels = labels[: self.max_seq_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_model_and_tokenizer(
    model_name,
    torch_dtype="bfloat16",
    device_map="auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16),
        device_map=device_map,
    )
    return model, tokenizer


def wrap_with_lora(model, lora_r, lora_alpha, lora_dropout):
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_stage1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--output_dir", type=str, default="out/qwen2.5-14b-prompt-enhancer-lora"
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch_dtype = "bfloat16" if use_bf16 else "float16"

    set_seed(args.seed)

    model, tokenizer = create_model_and_tokenizer(
        model_name=args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = wrap_with_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    shards = discover_shards(args.data_dir)
    train_rows = []
    for shard_path in shards:
        train_rows.extend(load_pairs_from_json(shard_path))
    train_dataset = PromptEnhancementDataset(
        train_rows, tokenizer, max_seq_length=args.max_seq_length
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=(not use_bf16),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
