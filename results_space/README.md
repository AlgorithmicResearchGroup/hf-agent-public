---
title: HF Agent Results
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# HF Agent Results

This Space renders a clean results page for reports written into a public Hugging Face Bucket by the multi-agent HF research runner.

Use it with query params:

- `bucket`: the bucket id, for example `username/hf-agent`
- `prefix`: the uploaded run prefix, for example `runs/20260327-084759-cyclefix`

Example:

`https://huggingface.co/spaces/<owner>/<space>?bucket=username/hf-agent&prefix=runs/20260327-084759-cyclefix`
