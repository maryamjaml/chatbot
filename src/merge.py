"""LoRA adapter ko base model mein merge karke ek standalone model save karta ha."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/chatbot-lora")
    ap.add_argument("--out", default="outputs/chatbot-merged")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged model save ho gaya: {args.out}")


if __name__ == "__main__":
    main()
