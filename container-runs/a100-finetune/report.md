# Executive Summary

**Recommendation (default): `Qwen/Qwen2.5-Coder-7B-Instruct`** for repo Q&A fine-tuning on **1× A100 80GB**, using **QLoRA/LoRA (PEFT) + TRL SFTTrainer**, and pairing with **RAG over your repository index**. It offers strong code-centric instruction-following, good long-context support for multi-file reasoning, broad tooling compatibility in Transformers (chat templates), and practical training/inference efficiency at 7B on a single A100.

If your use-case is heavy multi-file reasoning or you can tolerate higher latency/cost, step up to **`Qwen/Qwen2.5-Coder-32B-Instruct`**. If licensing/policy constraints make Qwen undesirable, a strong generalist alternative is **`meta-llama/Llama-3.1-8B-Instruct`** with code-focused SFT + strong retrieval.

# Recommended Model

## `Qwen/Qwen2.5-Coder-7B-Instruct`

- **Why this model for repo Q&A**: It is *code-specialized* (vs generalist instruct), so it tends to (a) answer code questions more precisely, (b) follow “show me where in the repo” instructions better, and (c) produce more usable patches/snippets.
- **Context length**: Qwen2.5 Coder instruct checkpoints are commonly used in **long-context** settings; verify the exact max context in the model card for your chosen revision and set `max_seq_len` accordingly.
- **Tooling support**: Works cleanly with **Transformers** `AutoTokenizer` / `AutoModelForCausalLM` and typically exposes a correct `tokenizer.chat_template`, which is critical for consistent SFT and inference formatting.
- **License**: Qwen models use the **Qwen license** (see the model card). Make sure its terms fit your distribution / internal-use requirements.

**Fine-tuning objective fit**: For “repository Q&A,” the strongest results usually come from:
1) RAG over a high-quality code index (paths, symbols, docstrings, READMEs, issues/PRs), plus
2) light SFT on your organization’s preferred answer style and repo-specific conventions.

# Alternatives and Tradeoffs

## 1) `Qwen/Qwen2.5-Coder-32B-Instruct` (higher quality, heavier)
- **Pick if**: you need the best multi-step reasoning across many files, stronger bug-finding, and better patch generation.
- **Tradeoffs**: higher latency, more complex serving, and finetuning is heavier (still feasible on 80GB with QLoRA, but expect lower throughput and tighter batch/sequence constraints).

## 2) `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (strong coder family; check integration)
- **Pick if**: you prefer DeepSeek-style coding behavior or see better performance on your internal evals.
- **Tradeoffs**: depending on the specific checkpoint, you may need `trust_remote_code` or additional compatibility checks; also confirm context length and license in the model card.

## 3) `meta-llama/Llama-3.1-8B-Instruct` (generalist, strong ecosystem)
- **Pick if**: licensing/compliance or org policy favors Llama, or you want a widely-supported base with many inference/quantization options.
- **Tradeoffs**: not code-specialized; you’ll lean more on **RAG quality** and **code-specific SFT data** to reach the same code precision as Qwen Coder.

# Hardware / Fine-Tuning Fit

Target hardware: **1× NVIDIA A100 80GB**.

**Practical fit** (typical approach):
- Use **LoRA / QLoRA** (PEFT) rather than full finetuning for speed and memory safety.
- Prefer **bf16** weights on A100; enable **FlashAttention-2** if installed; use **gradient checkpointing** for long context.
- For repo Q&A, long sequences (4k–16k) are often helpful; combine with **packing** to improve token throughput.

For 7B instruct models, this setup comfortably fits on 80GB with room for longer contexts and reasonable batch sizes. For 32B, QLoRA is typically required; expect smaller micro-batches and more careful sequence budgeting.

# Hugging Face Context

This workflow is “HF-native”:
- **Transformers** provides model/tokenizer loading and chat templates.
- **TRL** (`SFTTrainer`) provides supervised fine-tuning loops that work well for instruction tuning.
- **PEFT** provides LoRA/QLoRA adapters and easy save/load of lightweight deltas.
- For RAG, you can use **SentenceTransformers** for embeddings and **FAISS** for vector search, or swap in other HF embedding models.

# Hugging Face Models / Techniques

**Models (primary + alternatives)**
- `Qwen/Qwen2.5-Coder-7B-Instruct` (recommended)
- `Qwen/Qwen2.5-Coder-32B-Instruct` (upgrade option)
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (alternative)
- `meta-llama/Llama-3.1-8B-Instruct` (alternative)

**Techniques**
- **LoRA / QLoRA** via `peft.LoraConfig` + (optional) bitsandbytes 4-bit quantization.
- **Supervised fine-tuning (SFT)** via `trl.SFTTrainer` with `packing=True`.
- **Long-context efficiency**: FlashAttention-2 (`attn_implementation="flash_attention_2"`) + gradient checkpointing.
- **RAG for repo Q&A**: embeddings + FAISS retrieval; include file paths and retrieved snippets in the user message.

**Key HF libraries**
- `transformers`, `trl`, `peft`, `datasets`, `accelerate`

# Code Snippets

Below is a minimal, practical **LoRA SFT** script for repo Q&A instruction tuning using **TRL + PEFT**, formatted with the model’s **chat template**.

```python
#!/usr/bin/env python
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer

DEFAULT_SYSTEM = (
    "You are a helpful senior software engineer. Answer questions about the repository. "
    "Cite filenames/paths when relevant. If unsure, say what you would inspect next."
)


def build_messages(example):
    system = example.get("system") or DEFAULT_SYSTEM

    user_q = example["question"]
    meta = []
    for k in ("repo", "commit", "path"):
        if example.get(k):
            meta.append(f"{k}: {example[k]}")

    if example.get("context"):
        meta.append("retrieved_context:\n" + example["context"])

    if meta:
        user_q = user_q + "\n\n" + "\n".join(meta)

    answer = example["answer"]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_q},
        {"role": "assistant", "content": answer},
    ]


def format_chat(example, tokenizer):
    messages = build_messages(example)
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
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--use_flash_attn", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = None
    if args.use_flash_attn:
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

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
```

Line-by-line explanation:
1. Shebang to run as a script.
2. Import `argparse` for CLI flags.
3. Import `os` for creating the output directory.
4. Blank line for readability.
5. Import `torch` for dtype selection and model execution.
6. Import `load_dataset` to load JSONL training/eval data.
7. Import `LoraConfig` to configure PEFT LoRA adapters.
8. Import model/tokenizer loaders from Transformers.
9. Blank line.
10. Import TRL’s `SFTConfig` and `SFTTrainer` to run supervised fine-tuning.
11. Blank line.
12-15. Define a default system instruction for repo Q&A behavior.
16. Blank line.
17. Define `build_messages` to convert a dataset row into chat `messages`.
18. Choose per-example system prompt if provided, else the default.
19. Blank line.
20. Read the user question.
21. Create a list to hold optional metadata lines.
22-24. Add optional `repo/commit/path` metadata if present.
25-26. If retrieved context exists in the example, append it with a label.
27. Blank line.
28-29. If metadata exists, append it under the user question.
30. Blank line.
31. Read the assistant target answer.
32. Blank line.
33-37. Return a standard system/user/assistant message list.
38. Blank line.
39. Define `format_chat` to render messages into a single training string.
40. Build the message list.
41-46. Use the tokenizer’s chat template to produce the exact text format the model expects.
47. Return the updated example.
48. Blank line.
49. Define `main` entry point.
50. Create an argument parser.
51-54. Add required model/data/output arguments.
55. Blank line.
56-63. Add core training hyperparameters including long context and accumulation.
64. Blank line.
65-67. Add LoRA adapter hyperparameters.
68. Blank line.
69-70. Add performance flags for FlashAttention2 and gradient checkpointing.
71. Blank line.
72. Parse CLI arguments.
73. Blank line.
74. Create the output directory.
75. Blank line.
76. Load the tokenizer.
77-78. Ensure a pad token exists (decoder-only models often need this).
79. Blank line.
80. Initialize attention implementation to default.
81-82. If requested, switch to FlashAttention2.
83. Blank line.
84-90. Load the base model in bf16, place it on GPU, and optionally enable FlashAttention2.
91. Blank line.
92-94. Optionally enable gradient checkpointing and disable KV cache during training.
95. Blank line.
96-103. Load JSONL dataset splits for train (and optional eval).
104. Blank line.
105. Map the dataset into a single `text` field using the chat template.
106. Blank line.
107-126. Define the LoRA config and target common attention/MLP projection modules.
127. Blank line.
128-142. Define SFT configuration (packing, bf16, 8-bit optimizer, scheduler).
143. Blank line.
144-151. Create the SFT trainer with model, tokenizer, datasets, and LoRA config.
152. Blank line.
153. Run training.
154. Blank line.
155-156. Save the LoRA adapter weights and tokenizer.
157. Blank line.
158-159. Standard Python script entry point.

# Caveats / Open Questions

- **Verify context length & prompt format** for the exact model revision you deploy; set `max_seq_len` to match and ensure chat templates match between training and serving.
- **License/compliance**: confirm Qwen/DeepSeek/Llama license terms match your intended use and distribution.
- **Data quality dominates**: repo Q&A SFT datasets need clean questions, grounded answers, and correct file-path citations; noisy “hallucinated” labels will degrade behavior.
- **RAG design** matters: retrieval chunking, symbol-aware splitting, and adding file paths + line ranges usually improve grounding more than extra SFT.
- **Evaluation**: use a private set of repo questions with known answers; measure citation accuracy and “asked-to-open-file-next” behavior.

# References

- Qwen2.5 Coder 7B Instruct model repo: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Qwen2.5 Coder 32B Instruct model repo: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
- DeepSeek Coder V2 Lite Instruct repo: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- Llama 3.1 8B Instruct repo: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- TRL `SFTTrainer` docs: https://huggingface.co/docs/trl/en/sft_trainer
- PEFT (LoRA) docs: https://huggingface.co/docs/peft/en/index
- Transformers chat templates docs: https://huggingface.co/docs/transformers/en/chat_templating
