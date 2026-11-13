"""Model load karna aur jawab generate karna (chat.py aur app.py dono use karte hain)."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prepare_data import SYSTEM_PROMPT


def load_model(base: str, adapter: str | None = None):
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (
        torch.float16 if use_cuda else torch.float32
    )
    tok = AutoTokenizer.from_pretrained(adapter or base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map="auto" if use_cuda else None
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, messages, max_new_tokens=256, temperature=0.7, top_p=0.9):
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        repetition_penalty=1.1,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()
