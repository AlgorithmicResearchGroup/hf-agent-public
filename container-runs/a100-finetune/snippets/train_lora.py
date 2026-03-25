#!/usr/bin/env python
"""Minimal LoRA SFT skeleton for repo Q&A (Transformers + Datasets + PEFT + TRL).

Expected input JSONL (one per line) with fields:
  - system: str (optional)
  - user: str
  - assistant: str
Optionally include:
  - repo: str, path: str, commit: str, retrieved_context: str

Example row:
  {"system":"You are a repo assistant...","user":"Where is foo defined?","assistant":"It's in src/foo.py ..."}

Run:
  python snippets/train_lora.py \
    --model_id Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/train.jsonl \
    --eval_file data/eval.jsonl \
    --output_dir out/lora
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


DEFAULT_SYSTEM = (
    "You are a helpful codebase assistant. Answer repo questions using the provided context. "
    "Cite file paths and symbols. If unsure, ask a clarifying question."
)


def build_messages(example):
    system = example.get("system") or DEFAULT_SYSTEM

    user = example["user"]
    # Optional: add repo metadata or retrieved context into the user turn.
    meta = []
    for key in ("repo", "commit", "path"):
        if example.get(key):
            meta.append(f"{key}: {example[key]}")
    if example.get("retrieved_context"):
        meta.append("retrieved_context:\n" + example["retrieved_context"])

    if meta:
        user = user + "\n\n" + "\n".join(meta)

    assistant = example["assistant"]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def format_chat(example, tokenizer):
    messages = build_messages(example)
    # include the assistant content in the rendered text for supervised fine-tuning
    example["text"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return example


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)

    # LoRA knobs (good defaults from training_plan.md)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # performance knobs
    p.add_argument("--use_flash_attn", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    # Ensure pad token exists (common for decoder-only LMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = None
    if args.use_flash_attn:
        # Requires flash-attn installed and a compatible GPU/driver.
        attn_impl = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            **({"eval": args.eval_file} if args.eval_file else {}),
        },
    )

    ds = ds.map(lambda ex: format_chat(ex, tokenizer), remove_columns=ds["train"].column_names)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_len,
        packing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        peft_config=lora_config,
        args=sft_config,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer.
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
