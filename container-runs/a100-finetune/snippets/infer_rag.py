#!/usr/bin/env python
"""Inference + tiny RAG (FAISS) for repo Q&A.

This script:
  1) loads a base chat model + optional LoRA adapter
  2) embeds a small set of repo chunks (from a JSONL)
  3) retrieves top-k chunks for a query
  4) sends (query + retrieved context) to the chat model

Input chunks JSONL format (one per line):
  {"path":"src/foo.py","text":"...chunk text..."}

Run:
  python snippets/infer_rag.py \
    --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter_dir out/lora \
    --chunks_file data/chunks.jsonl \
    --question "Where is Foo implemented?"
"""

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

    # Embedding model: pick a strong general model; swap for code-specific if desired.
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
