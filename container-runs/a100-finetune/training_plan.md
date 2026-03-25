# Training plan: repo Q&A fine-tuning on **1× A100 80GB** (hardware + strategy)

This memo maps a single A100 80GB to realistic fine-tuning paths for **repository Q&A** (codebase-grounded questions, API/behavior queries, “where is X implemented?”, “why does test Y fail?”). It gives **2–3 viable recipes**, concrete hyperparameters, and operational limits for training + serving on the same GPU.

---

## 0) Hardware facts that matter (A100 80GB)

**A100 80GB (SXM/PCIe)** comfortably supports:
- **bf16** training/inference (recommended)
- **FlashAttention / SDPA** (big throughput win at long context)
- **LoRA/QLoRA** for 14B–34B class models
- **Full fine-tune** for ~7B–14B class models at moderate sequence length with checkpointing

### Memory rule-of-thumb (bf16)
- **Weights**: ~2 bytes/parameter.
  - 7B ≈ 14 GB
  - 14B ≈ 28 GB
  - 34B ≈ 68 GB
- **Optimizer states (AdamW)**: ~8–12 bytes/parameter depending on implementation (m, v, fp32 master weights etc.). This is why **full fine-tune** becomes expensive quickly.
- **Activations**: scales with **batch × seq_len × hidden**; mitigated by **gradient checkpointing** and FlashAttention.

Practical implication: 
- Full FT is easy at **7B**, feasible-but-tight at **14B**, and generally impractical at **≥34B** on one GPU.
- LoRA/QLoRA makes **14B–34B** easy; context length becomes the main memory driver.

---

## 1) Path A — Full fine-tune a smaller model (7B–14B)

### When to choose
- You need **maximum behavioral shift** (tone, formatting, tool-use) and can afford longer training.
- Dataset is large and high-quality (10^5–10^6 examples) and you want best downstream generalization.

### Recommended target
- **7B** default for full FT on 1×A100 unless you *must* push quality: 7B gives ample headroom for long context and higher throughput.
- **14B** feasible with more constraints (shorter context or lower batch + more checkpointing).

### Concrete recipe (7B full FT)
- Precision: **bf16**
- Attention: **FlashAttention2** if available; otherwise PyTorch **SDPA**
- Optimizer: **AdamW** (or **AdamW8bit** if you want more memory headroom)
- LR schedule: cosine, warmup 3–5%
- Gradient checkpointing: **ON**

**SFT hyperparameters (starting point):**
- `max_seq_len`: **4096** (increase to 8192 if your base model supports it and data benefits)
- `micro_batch_size`: **1–2**
- `grad_accum`: **16–32** (target global batch 32–64 sequences)
- `learning_rate`: **1e-5 to 2e-5** (full FT is sensitive)
- `weight_decay`: **0.1**
- `max_grad_norm`: **1.0**
- `epochs`: **1–2** (more if dataset is small)

**Estimated fit (7B, seq 4k):**
- Should fit in 80GB with checkpointing.
- Throughput depends on implementation; as a rough planning number: **~20–60 tokens/s** training-step effective, yielding **hours to a couple days** depending on dataset size and sequence length.

### Concrete recipe (14B full FT, “tight but doable”)
Use only if you accept slower training and reduced batch/context.
- `max_seq_len`: **2048–4096**
- `micro_batch_size`: **1**
- `grad_accum`: **32–64**
- LR: **~1e-5**
- Must use: **gradient checkpointing + FlashAttention/SDPA**

**Operational caveat:** full FT increases risk of regressions; keep a strong eval harness and frequent checkpoints.

---

## 2) Path B — LoRA / QLoRA on 14B–34B class (recommended default)

### When to choose
- You want strong results fast with controllable memory.
- Repo Q&A often benefits from **instruction-following + grounding patterns** more than changing core knowledge.
- You want to preserve base model coding ability while adding repo-specific behaviors.

### Recommendation (default)
- **LoRA in bf16** for 14B (highest quality/throughput trade-off).
- **QLoRA (4-bit)** for 34B if you want a larger base model while staying within 80GB.

### LoRA config (good default)
- Target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `r`: **16–64** (start 32)
- `lora_alpha`: **2×r** (e.g., 64)
- `lora_dropout`: **0.05**

### SFT hyperparameters (LoRA)
- Precision: **bf16**
- Optimizer: **paged_adamw_8bit** (bitsandbytes) or AdamW
- `learning_rate`: **1e-4** (LoRA tolerates higher LR)
- `max_seq_len`: **8192** (repo Q&A benefits from long context: files/snippets)
- `micro_batch_size`: **1–2**
- `grad_accum`: **8–32** (target global batch 16–64)
- `epochs`: **2–4** for small/medium datasets; **1–2** for large

### Memory/fit guidance
- **14B LoRA (bf16)**: should easily fit with **8k context** with checkpointing.
- **34B QLoRA (4-bit)**: typically fits in 80GB; context length may need to stay **≤4096–8192** depending on implementation.

### Throughput expectations
- LoRA is usually compute-bound; QLoRA adds some overhead.
- Planning numbers:
  - **14B LoRA @8k**: expect **hours–1 day** for ~50k–200k examples depending on packing and average length.
  - **34B QLoRA @4k–8k**: expect **~1–3 days** for similar token counts.

### Best practices
- Use **example packing** (concatenate multiple short Q&As) to improve GPU utilization.
- Keep **validation set** with repo questions requiring retrieval (to prevent overfitting to synthetic artifacts).

---

## 3) Path C — Preference tuning (DPO/ORPO) after SFT

### When to choose
- You have (or can generate) **pairwise preferences**: good vs bad answers.
- You want to enforce behaviors like:
  - cite files/paths
  - admit uncertainty / ask clarifying questions
  - avoid hallucinating non-existent symbols
  - produce patch-style diffs with correct formatting

### Recommended pipeline
1. **SFT (LoRA/QLoRA)** on instruction + repo QA
2. **Preference tuning** (DPO or ORPO) using pairs from:
   - model-vs-model comparisons
   - model outputs judged by heuristics (e.g., references existing files) + limited human review

### DPO/ORPO hyperparameters (starting point)
- Method: **DPO** (strong baseline) or **ORPO** (simpler training loop)
- `beta`: **0.05–0.2** (start 0.1)
- LR: **5e-6 to 2e-5** (smaller than SFT)
- Batch: same as SFT; prefer **shorter seq_len (2k–4k)** unless you have long preference samples
- Steps: **1–5k** updates is often enough to noticeably improve refusal/grounding style.

### Memory/fit
- DPO runs two forward passes per batch (chosen/rejected); plan **~1.6–2×** compute cost vs SFT.

---

## 4) Data strategy (repo Q&A + grounding)

### Core training mixture (recommended)
- **Instruction SFT (code/general)**: keep base coding competence.
  - 20–40% of tokens
- **Repo Q&A (real)**: issues/PR discussions, internal docs, tests, READMEs.
  - 30–50%
- **Synthetic repo QA (grounded)**: generated questions from code chunks, docstrings, and file trees.
  - 20–40%

### How to generate synthetic repo QA (high-signal)
- Sample a file chunk (e.g., 200–800 lines) + metadata (path, module name).
- Ask model to produce:
  - “where is X implemented?”
  - “what does function Y do, edge cases?”
  - “how to call API Z?”
  - “write a minimal repro / unit test”
- **Critical constraint:** answers must quote relevant snippet and reference correct file paths.

### RAG vs pure fine-tune (recommend: both)
- **RAG** is essential for:
  - fast-changing repos
  - large codebases
  - long-tail details
- **Fine-tuning** is best for:
  - interaction patterns (ask for repo version, clarify branch)
  - citation discipline (always include file paths)
  - formatting patches, command snippets, test instructions

**Practical hybrid:**
- At inference time, retrieve top-k chunks (e.g., k=10–30), rerank, and feed into the model.
- Fine-tune on *RAG-style inputs* so the model learns to use retrieved context.

---

## 5) Evaluation plan (RepoBench-like + practical)

### Automatic evaluations
1. **Retrieval-grounded QA accuracy**
   - Questions whose answer exists in repo; score exact-match on file path/function name + semantic similarity.
2. **Citation correctness**
   - Percentage of answers that cite existing paths and symbols.
3. **No-hallucination checks**
   - Detect references to non-existent files/symbols (static scan).
4. **Patch validity**
   - For diff outputs: applies cleanly + tests pass on a fixed commit.

### Human spot checks (high ROI)
- 50–200 questions sampled across:
  - easy doc questions
  - “trace behavior across modules”
  - bug triage
- Rate: correctness, groundedness, helpfulness, and whether model asks clarifying questions.

### Safety/refusal constraints
- Enforce policies for secrets, credentials, and license constraints.
- Preference tuning pairs should include:
  - refusal to exfiltrate secrets
  - safe handling of security vulnerabilities (responsible disclosure style)

---

## 6) Concrete recommended training recipe (default)

### Default pick for 1×A100 80GB
**LoRA SFT on a 14B-class coding/instruction model**, then optional **DPO/ORPO**.

Rationale: best balance of quality, long context, and speed on a single GPU; avoids the heavy optimizer memory of full FT.

### Phase 1: SFT (LoRA)
- Context: **8192**
- Packing: **ON**
- Precision: **bf16**
- Attention: FlashAttention2/SDPA
- Checkpointing: ON

Suggested hyperparameters:
- `max_seq_len`: **8192**
- `micro_batch_size`: **1**
- `gradient_accumulation_steps`: **16** (global batch 16 sequences)
- `lr`: **1e-4**
- `warmup_ratio`: **0.03**
- `epochs`: **2** (or stop by val loss + eval metrics)
- `lora_r`: **32**, `lora_alpha`: **64**, `dropout`: **0.05**

### Phase 2: Preference tuning (DPO or ORPO)
- Context: **2048–4096**
- `lr`: **1e-5**
- `beta`: **0.1**
- Steps: **1k–5k**

### Expected runtime (order-of-magnitude)
Depends mainly on **total training tokens**.
- For ~100M tokens at 8k context with packing, plan **~8–24 hours** LoRA SFT.
- Preference tuning on ~5–20M tokens: **~2–8 hours**.

(These are planning estimates; actual throughput varies with kernels, model family, and data packing.)

---

## 7) Serving on the same GPU: operational limits

If you plan to train and serve on the **same** A100, you generally can’t do both simultaneously.

After training, for inference:
- **14B bf16**: typically fine, leaving room for KV cache at moderate concurrency.
- **34B 4-bit**: fits well; may have lower throughput.

Serving constraints driven by KV cache:
- KV cache grows with **batch × seq_len × layers × hidden**.
- Practical knobs:
  - cap context at **8k** (or 4k for higher concurrency)
  - limit concurrent requests
  - use **paged attention**/vLLM if available

---

## 8) Quick decision table

| Path | Model scale | Method | Fits on 1×A100 80GB | Best for |
|---|---:|---|---|---|
| A | 7B | Full FT | Yes (comfortable) | max behavior change, high throughput |
| A’ | 14B | Full FT | Borderline / careful | quality bump if willing to trade speed |
| B (default) | 14B | LoRA bf16 | Yes (8k context feasible) | best overall trade-off |
| B’ | 34B | QLoRA 4-bit | Yes | bigger base model under memory constraints |
| C | any | DPO/ORPO after SFT | Yes | grounding/refusal/format compliance |
