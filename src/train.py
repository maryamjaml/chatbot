"""LoRA/QLoRA fine-tuning (Hugging Face TRL + PEFT)."""
import argparse
import math

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_file", default="data/processed/train.jsonl")
    ap.add_argument("--val_file", default="data/processed/val.jsonl")
    ap.add_argument("--output_dir", default="outputs/chatbot-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA: kam VRAM mein train")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        quantization_config=quant,
        device_map="auto" if use_cuda else None,
    )
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        gradient_checkpointing=True,
        max_length=args.max_len,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Val loss: {metrics['eval_loss']:.4f} | perplexity: {math.exp(metrics['eval_loss']):.2f}")

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Adapter save ho gaya: {args.output_dir}")


if __name__ == "__main__":
    main()
