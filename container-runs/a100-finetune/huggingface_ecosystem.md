# Hugging Face landscape for Repo Q&A / code-chat (open-weight)

This memo shortlists strong *open-weight* models on the Hugging Face Hub that are practical bases for instruction-tuning into a **“chat with your codebase / repository Q&A”** assistant (often paired with RAG over a code index).

> Scope: model scouting + key HF ecosystem assets (datasets/evals/examples). Hardware/fine-tuning strategy is handled by teammate `training`.

## 1) Shortlist: models worth considering

| Model (HF repo) | Params | Context length | License | Quantized variants on HF? | Chat template / notes | Repo Q&A / RAG suitability notes |
|---|---:|---:|---|---|---|---|
| **Qwen/Qwen2.5-Coder-7B-Instruct** https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct | 7B | (model card; typically long-context for Qwen2.5 Coder—verify per card) | Qwen license (see model card) | Yes (e.g., official **GGUF** repo: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF; many community AWQ/GGUF) | Uses Qwen chat template (`tokenizer.chat_template` in Transformers). Strong code/chat behavior. | Very strong default for code-centric chat. Pair with retrieval over code chunks + file paths; good at editing + explaining. |
| **Qwen/Qwen2.5-Coder-32B-Instruct** https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct | 32B | (verify per card) | Qwen license (see model card) | Yes (community quantizations common; GGUF exists for other sizes; check hub) | Same Qwen template. Higher reasoning/code quality; heavier inference. | Great for “hard” repo Q&A and multi-file reasoning; excellent when you can afford VRAM/latency. |
| **deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct** https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | Lite | (verify per card) | DeepSeek license (see card) | Yes (e.g. GGUF community repos) | May require `trust_remote_code` depending on Transformers support; check model card. | Strong coding + long-context family; good option if you want DeepSeek-style code reasoning at smaller footprint. |
| **deepseek-ai/deepseek-coder-6.7b-instruct** https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct | 6.7B | (verify per card) | DeepSeek license (see card) | Yes (community AWQ/GGUF common) | Llama-style-ish tokenizer/config; check card for prompt format. | Solid baseline for code chat; older than V2 but widely used and easy to run. |
| **meta-llama/Llama-3.1-8B-Instruct** https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | 8B | (verify per card; Llama 3.1 supports long context) | Llama 3.1 community license | Yes (e.g. AWQ INT4: https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 ; GGUF available) | Uses Llama 3.1 instruct chat template; best generalist instruct baseline. | Not code-specialized but strong general reasoning; good for “repo Q&A” when paired with high-quality retrieval + code-oriented SFT. |
| **bigcode/starcoder2-15b** (base) https://huggingface.co/bigcode/starcoder2-15b and instruct finetunes (e.g. TechxGenus/starcoder2-15b-instruct) https://huggingface.co/TechxGenus/starcoder2-15b-instruct | 15B | (verify per card) | BigCode OpenRAIL-M | Yes (GGUF community repos exist) | Instruct variants are community; check prompt format in README. | Strong code completion lineage. For repo Q&A you usually want an *instruct-tuned* variant; quality varies by finetune. |

### Notes on “Codestral” / Mistral code models
Some Mistral-derived code models may have restricted distribution or may not appear as first-class HF repos depending on release policy at the time you run this. If you have access to an official repo on HF, include it as an additional candidate; otherwise prefer Qwen/DeepSeek/StarCoder2/Llama.

## 2) Ecosystem assets for “chat with your repo”

### Common RAG + code indexing building blocks (HF-friendly)
- **SentenceTransformers / embedding models**: many teams use code embeddings for retrieval (e.g., `BAAI/bge-*`, `intfloat/e5-*`, and code-specialized embedding models). For codebases, choose an embedding model that handles code tokens well.
- **Transformers chat templates**: prefer models with an official `tokenizer.chat_template` so your fine-tuning/inference formatting is consistent.
- **Quantization formats you’ll encounter on HF**:
  - **GGUF** repos for llama.cpp inference.
  - **AWQ / GPTQ** repos for fast 4-bit inference on GPUs.
  - **bitsandbytes 4-bit (NF4)** for QLoRA-style finetuning.

### Datasets to know (code pretraining, instruction, and eval)
- **bigcode/the-stack** (pretraining-scale code) — often cited as core data for code LMs.
- **code_search_net** (CodeSearchNet) — classic code search dataset (NL↔code), useful for retrieval evaluation baselines.
- **RepoBench** — evaluates repository-level/codebase understanding and multi-file reasoning.
- **SWE-bench** — evaluates agentic software engineering (bugfixing in real repos); used widely as a “realistic” benchmark for repo understanding + patching.

### Evaluation harnesses / tooling
- **lm-evaluation-harness** (EleutherAI) — general LLM eval harness; some code tasks integrated.
- **bigcode-evaluation-harness** — code-centric evaluation harness used in the BigCode project.

### HF examples / Spaces to look for
Search HF Spaces / repos with keywords:
- “repository qa”, “repo chat”, “codebase chat”, “rag over code”, “chat with your code”, “swe-bench”.

(You can quickly discover these via Hub search; prioritize Spaces demonstrating **RAG over a GitHub repo** + a code LLM.)

## 3) Practical model-selection guidance (quick)

- **Best default code-chat base**: **Qwen2.5-Coder-Instruct** (7B for speed; 32B for quality).
- **Strong alternative code family**: **DeepSeek-Coder** (V2 Lite Instruct if you want newer family; 6.7B Instruct as a common baseline).
- **Generalist baseline**: **Llama-3.1-8B-Instruct** (works well for Q&A; add code SFT and retrieval to compensate for lack of code specialization).
- **StarCoder2**: best when you want BigCode lineage + OpenRAIL-M terms; ensure you choose a well-regarded instruct variant.

## References (starting set; expand/verify from model cards)
1. Qwen/Qwen2.5-Coder-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
2. Qwen/Qwen2.5-Coder-7B-Instruct-GGUF — https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
3. Qwen/Qwen2.5-Coder-32B-Instruct — https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
4. deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct — https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
5. deepseek-ai/deepseek-coder-6.7b-instruct — https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct
6. meta-llama/Llama-3.1-8B-Instruct — https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
7. hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 — https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
8. bigcode/starcoder2-15b — https://huggingface.co/bigcode/starcoder2-15b
9. TechxGenus/starcoder2-15b-instruct — https://huggingface.co/TechxGenus/starcoder2-15b-instruct
