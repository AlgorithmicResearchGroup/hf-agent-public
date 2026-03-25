# Hugging Face Model Advisor

A multi-agent Hugging Face model-selection and fine-tuning system that investigates current options, searches the web and Hugging Face, and produces Markdown recommendation reports with citations and code snippets.

## What Changed

This repo is now focused on Hugging Face model-selection and fine-tuning workflows.

- The old market-research feasibility prompts and tools are removed.
- The multi-agent orchestrator shape remains.
- The primary remote execution path is Hugging Face Jobs.
- A typical task is: `"I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"`

## Setup

```bash
pip install -r requirements.txt
```

Set the secrets you need:

```bash
export OPENAI=...
export TAVILY_API_KEY=...
```

Optional:

```bash
export ANTHROPIC=...
export SETTINGS_PACK=default
```

## Run Locally

Single agent:

```bash
python main.py "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
```

Multi-agent orchestrator:

```bash
python run_collab_long.py "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
```

Prompt file:

```bash
python run_collab_long.py --prompts challenge_prompts.json --prompt 1
```

## Run on Hugging Face Jobs

This repo includes a GitHub Actions workflow at `.github/workflows/publish.yaml` that builds and publishes the Docker image to GHCR on pushes to `main`/`master`, on version tags, and on manual dispatch.

The published image path is:

```bash
ghcr.io/algorithmicresearchgroup/hf-agent-public:latest
```

You can still build locally if needed:

```bash
docker build -t ghcr.io/algorithmicresearchgroup/hf-agent-public:latest .
```

Then point the launcher at that image:

```bash
export HF_JOB_IMAGE=ghcr.io/algorithmicresearchgroup/hf-agent-public:latest
```

Launch the full orchestrator as one remote job:

```bash
python launch_hf_job.py "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
```

Wait for completion and stream final logs:

```bash
python launch_hf_job.py "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?" --wait
```

The launcher submits one Hugging Face Job using your prebuilt Docker image and runs `python run_collab_long.py "<query>"` inside that image.

## Tooling

Base advisor tools:

- `search_web`
- `fetch_page`
- `search_huggingface`
- `inspect_huggingface_repo`
- `read_huggingface_file`
- `run_bash`
- `read_file`
- `write_file`
- `edit_file`

When protocol mode is enabled, agents also use:

- `send_message`
- `check_messages`

## Output Contract

The orchestrator aims to produce `report.md` with:

- Executive Summary
- Recommended Model
- Alternatives and Tradeoffs
- Hardware / Fine-Tuning Fit
- Hugging Face Context
- Hugging Face Models / Techniques
- Code Snippets
- Caveats / Open Questions
- References

The final report is expected to embed its code snippets directly in `report.md`, with a line-by-line explanation for each embedded snippet.

If the query is about a model family, architecture, optimization, kernel, or technique, the report should explicitly name the relevant Hugging Face models, repos, libraries, docs, datasets, or Spaces instead of referring to Hugging Face only generically.

If the query includes hardware or budget constraints, the report should turn them into a concrete recommendation about model scale, fine-tuning method, and operational caveats.

Intermediate files commonly include `papers.md`, `huggingface_ecosystem.md`, `code_examples.md`, and optional `snippets/`.

## Notes

- The current Flask web UI is not the primary path and has not been expanded for the new workflow.
- `run_collab_long.py` is the core orchestrator and is the command that the Hugging Face Job runs remotely.
