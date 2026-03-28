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
export SETTINGS_PACK=litellm_openai
```

Provider packs in `manifest.json`:

- `litellm_openai` uses LiteLLM with `openai/gpt-5.2`
- `litellm_anthropic` uses LiteLLM with `anthropic/claude-opus-4-6`
- `default` keeps the direct OpenAI SDK path
- `anthropic` keeps the direct Anthropic SDK path

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

For self-serve report delivery, also choose a Bucket you own:

```bash
hf buckets create your-username/hf-agent
```

If you want a clean user-facing results page instead of raw bucket links, deploy the companion Space in [`results_space/`](./results_space) and note its public URL:

```bash
export RESULTS_SPACE_URL=https://huggingface.co/spaces/your-username/hf-agent-results
```

Launch the full orchestrator as one remote job:

```bash
python launch_hf_job.py \
  --report-bucket your-username/hf-agent \
  --results-space-url "$RESULTS_SPACE_URL" \
  "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
```

Wait for completion and stream final logs:

```bash
python launch_hf_job.py \
  --report-bucket your-username/hf-agent \
  --report-prefix runs/my-query \
  --results-space-url "$RESULTS_SPACE_URL" \
  --wait \
  "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
```

The launcher submits one Hugging Face Job using your prebuilt Docker image and runs `python run_collab_long.py "<query>"` inside that image.
If `--report-bucket` is set, the job uploads `report.md`, `run_summary.json`, `qa_report.json`, and related artifacts to that bucket.
The launcher and remote job both print a `RESULT LINKS` block with:

- the optional Space results URL
- the direct `report.md` viewer URL
- the direct download URL
- the bucket folder URL

When `--report-prefix` is omitted, the launcher now generates one up front so those links are known immediately after job submission.

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
When bucket upload is enabled, the run also writes:

- `artifacts_manifest.json` for the full artifact inventory
- `delivery.json` for the human-facing result links

The uploaded `run_summary.json` also records the direct viewer/download/folder URLs.

## Notes

- The current Flask web UI is not the primary path and has not been expanded for the new workflow.
- `run_collab_long.py` is the core orchestrator and is the command that the Hugging Face Job runs remotely.
- [`results_space/`](./results_space) is a small deployable companion Space that renders a clean results page from a public Bucket `delivery.json`.
