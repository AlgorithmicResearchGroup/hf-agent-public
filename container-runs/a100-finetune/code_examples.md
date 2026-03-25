# Code examples (repo Q&A LoRA SFT + RAG inference)

These snippets are **minimal, runnable skeletons** aligned with the team’s plan (A100 80GB; LoRA SFT; RAG at inference). They assume a chat model with a valid `tokenizer.chat_template` (e.g., **Qwen2.5 Coder Instruct**).

## Environment / installs

```bash
# Core
pip install -U "transformers>=4.41" "datasets>=2.19" "accelerate>=0.31" "peft>=0.11" "trl>=0.9" \
  "bitsandbytes>=0.43" "sentence-transformers>=3.0" "faiss-cpu>=1.8"

# If you want GPU FAISS instead:
# pip install -U faiss-gpu

# Optional speedups on A100 (requires a matching CUDA toolchain)
# pip install -U flash-attn --no-build-isolation
```

Notes:
- For best throughput on A100: prefer **bf16**, enable **FlashAttention2** when available (`--use_flash_attn`), and use **gradient checkpointing** for long context.
- If FlashAttention2 isn’t installed, Transformers will fall back to SDPA/eager attention.

---

## Snippet 1 — LoRA SFT training (TRL `SFTTrainer`)

File: `snippets/train_lora.py`

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
    "You are a helpful codebase assistant. Answer repo questions using the provided context. "
    "Cite file paths and symbols. If unsure, ask a clarifying question."
)


def build_messages(example):
    system = example.get("system") or DEFAULT_SYSTEM

    user = example["user"]
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

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
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

---

## Snippet 2 — Inference with simple RAG (SentenceTransformers + FAISS)

File: `snippets/infer_rag.py`

```python
#!/usr/bin/env python
import argparse
import json

import faiss
import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM = (
    "You are a helpful repo Q&A assistant. Use retrieved context; cite file paths. "
    "If the answer isn't in context, say what you would search next."
)


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_faiss(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def retrieve(index, embedder, chunks, question, k=8):
    q = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    results = [chunks[i] for i in idxs[0].tolist()]
    return results


def format_context(results):
    ctx_parts = []
    for r in results:
        ctx_parts.append(f"FILE: {r.get('path','(unknown)')}\n{r['text']}")
    return "\n\n---\n\n".join(ctx_parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--chunks_file", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    args = p.parse_args()

    chunks = load_chunks(args.chunks_file)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True)
    index = build_faiss(emb)

    retrieved = retrieve(index, embedder, chunks, args.question, k=args.top_k)
    retrieved_context = format_context(retrieved)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {args.question}\n\n"
                f"Retrieved context (top-{args.top_k}):\n{retrieved_context}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
```

Line-by-line explanation:
1. Shebang to run as a script.
2. Import `argparse` for CLI flags.
3. Import `json` to parse JSONL chunks.
4. Blank line.
5. Import FAISS for vector search.
6. Import NumPy for embedding arrays.
7. Import PyTorch for model execution and dtype.
8. Import `PeftModel` to load the LoRA adapter on top of the base model.
9. Import `SentenceTransformer` to embed code/text chunks.
10. Import model/tokenizer loaders from Transformers.
11. Blank line.
12-15. Define a system instruction emphasizing grounding and citations.
16. Blank line.
17. Define `load_chunks` to read chunk JSONL into Python dicts.
18. Initialize the chunk list.
19-22. Read the file line-by-line and append parsed JSON objects.
23. Return the list of chunks.
24. Blank line.
25. Define `build_faiss` to create a cosine-similarity FAISS index.
26. Read embedding dimensionality.
27. Create an inner-product FAISS index.
28. Normalize vectors so inner product equals cosine similarity.
29. Add vectors to the index.
30. Return the index.
31. Blank line.
32. Define `retrieve` to fetch top-k chunks for the question.
33. Embed the question into a vector.
34. Normalize the query for cosine similarity.
35. Search the FAISS index.
36. Convert indices into chunk dicts.
37. Return the retrieved chunks.
38. Blank line.
39. Define `format_context` to pack retrieved chunks into a single context string.
40. Initialize context parts list.
41-42. Add each chunk with a FILE header containing its path.
43. Join chunks with separators.
44. Blank line.
45. Define `main` entry point.
46. Create an argument parser.
47-52. Add CLI args for base model, adapter, chunks, question, and generation.
53. Parse args.
54. Blank line.
55. Load repository chunks.
56. Blank line.
57. Load an embedding model (swap to a code embedding model as needed).
58. Embed all chunks.
59. Build a FAISS index.
60. Blank line.
61. Retrieve top-k chunks for the question.
62. Format them into a single string.
63. Blank line.
64. Load the base model tokenizer.
65-66. Ensure a pad token exists.
67. Blank line.
68-72. Load the base chat model in bf16 on GPU.
73. Blank line.
74-75. If provided, load the LoRA adapter on top of the base model.
76. Blank line.
77-91. Build chat `messages` containing the question and retrieved context.
92. Blank line.
93-97. Render the messages into a prompt, adding a generation marker.
98. Blank line.
99. Tokenize and move tensors to the model device.
100-109. Generate deterministically (greedy) with a max token budget.
110. Blank line.
111-112. Decode and print the final text.
113. Blank line.
114-115. Standard Python script entry point.

---

## Suggested dataset format (JSONL)

Training rows (JSONL):
```json
{"system":"You are a repo assistant...","user":"What does `FooBar` do?","assistant":"`FooBar` ... (cite path)"}
```

Optional “RAG-style” fields you can add during data creation (recommended so the model learns to use retrieved context):
```json
{"user":"How does auth work?","retrieved_context":"FILE: src/auth.py\n...","assistant":"Auth flow: ..."}
```
