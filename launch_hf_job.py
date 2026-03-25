#!/usr/bin/env python3
"""
Launch the full multi-agent orchestrator as a single Hugging Face Job.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional

from huggingface_hub import HfApi, run_job


DEFAULT_SECRET_NAMES = [
    "OPENAI",
    "ANTHROPIC",
    "TAVILY_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
]

DEFAULT_ENV_NAMES = [
    "SETTINGS_PACK",
    "PROVIDER",
    "AGENT_MODEL",
    "ORCHESTRATOR_MODEL",
    "MAX_ITERATIONS",
    "MAX_RUNTIME_SECONDS",
    "MAX_WAVES",
    "QA_ITERATIONS",
    "FIX_ITERATIONS",
    "FIX_RUNTIME_SECONDS",
]

DEFAULT_IMAGE_ENV = "HF_JOB_IMAGE"


def collect_existing_env(names) -> Dict[str, str]:
    env = {}
    for name in names:
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def resolve_job_image(cli_value: Optional[str]) -> Optional[str]:
    return cli_value or os.environ.get(DEFAULT_IMAGE_ENV)


def build_remote_command(query: str) -> list[str]:
    return ["python", "run_collab_long.py", query]


def wait_for_job(job_id: str, namespace: Optional[str], poll_interval: int) -> str:
    api = HfApi()
    while True:
        job = api.inspect_job(job_id=job_id, namespace=namespace)
        stage = str(job.status.stage)
        print(f"[HF JOB] {job.id} stage={stage}", flush=True)
        if stage in {"COMPLETED", "ERROR", "CANCELED", "DELETED"}:
            print("[HF JOB] Final logs:", flush=True)
            for line in api.fetch_job_logs(job_id=job_id, namespace=namespace):
                print(line, end="" if line.endswith("\n") else "\n")
            return stage
        time.sleep(poll_interval)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Hugging Face model advisor orchestrator as a Hugging Face Job")
    parser.add_argument("query", help="Model-selection or fine-tuning query to run remotely")
    parser.add_argument(
        "--image",
        help=f"Docker image for the job. Defaults to the {DEFAULT_IMAGE_ENV} environment variable.",
    )
    parser.add_argument("--flavor", default="cpu-basic", help="Hugging Face Jobs hardware flavor")
    parser.add_argument("--timeout", default="2h", help="Maximum job duration")
    parser.add_argument("--namespace", help="Optional Hugging Face user or org namespace")
    parser.add_argument("--wait", action="store_true", help="Wait for the remote job to finish")
    parser.add_argument("--poll-interval", type=int, default=15, help="Seconds between status polls when waiting")
    return parser.parse_args()


def main():
    args = parse_args()
    image = resolve_job_image(args.image)
    if not image:
        raise SystemExit(
            f"Set --image or define {DEFAULT_IMAGE_ENV} to a prebuilt Docker image that contains this repo."
        )

    env = {"PYTHONUNBUFFERED": "1", **collect_existing_env(DEFAULT_ENV_NAMES)}
    secrets = collect_existing_env(DEFAULT_SECRET_NAMES)

    job = run_job(
        image=image,
        command=build_remote_command(args.query),
        env=env,
        secrets=secrets or None,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
    )

    print(f"Job started: {job.id}")
    print(f"URL: {job.url}")
    print(f"Flavor: {job.flavor}")
    print(f"Image: {image}")

    if not args.wait:
        return

    stage = wait_for_job(job.id, args.namespace, args.poll_interval)
    sys.exit(0 if stage == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
