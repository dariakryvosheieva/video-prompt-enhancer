import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from train_stage1 import INSTRUCTION_TEXT


ADAPTER_DIR = "out/qwen2.5-14b-prompt-enhancer-lora-stage2"


@torch.inference_mode()
def qualitative_eval(
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    device_map="auto",
    dtype=None,
):
    if dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
    base_model_name = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    formatted = f"{INSTRUCTION_TEXT}\n\n" f"Input:\n{prompt.strip()}\n\n" "Output:\n"

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = generated[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


if __name__ == "__main__":
    prompt = "microraptor gliding from one tree branch to another"
    print(qualitative_eval(prompt))
