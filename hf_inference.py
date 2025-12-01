import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-14B-Instruct"
adapter_id = "dariakryvosheieva/video-prompt-enhancer"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", torch_dtype="auto"
)
model = PeftModel.from_pretrained(base, adapter_id).eval()


def format_query(simple_prompt: str) -> str:
    instruction_text = (
        "Convert the following video generation prompt into a professional-grade prompt that will produce a high quality, aesthetic, and impressive video."
        "If the original prompt includes a style specification (such as 'anime', 'pixel', or 'cartoon'), keep it in the converted prompt."
        "Output only the converted prompt."
    )
    return f"{instruction_text}\n\nInput:\n{simple_prompt.strip()}\n\nOutput:\n"


prompt = "a cat riding a skateboard in a park at sunset"
text = format_query(prompt)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )

print(
    tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
)
