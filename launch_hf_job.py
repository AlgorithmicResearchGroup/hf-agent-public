#!/usr/bin/env python3
"""
Launch the full multi-agent orchestrator as a single Hugging Face Job.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Optional

import requests
from huggingface_hub import HfApi, run_job

from artifact_publisher import (
    build_results_page_url,
    bucket_file_resolve_url,
    bucket_file_view_url,
    bucket_folder_url,
    normalize_report_prefix,
)


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
    "REPORT_BUCKET",
    "REPORT_PREFIX",
    "RESULTS_SPACE_URL",
]

DEFAULT_IMAGE_ENV = "HF_JOB_IMAGE"
HF_ENDPOINT = "https://huggingface.co"


def collect_existing_env(names) -> Dict[str, str]:
    env = {}
    for name in names:
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def resolve_job_image(cli_value: Optional[str]) -> Optional[str]:
    return cli_value or os.environ.get(DEFAULT_IMAGE_ENV)


def resolve_results_space_url(cli_value: Optional[str]) -> Optional[str]:
    if not cli_value:
        return None
    value = cli_value.rstrip("/")
    marker = f"{HF_ENDPOINT}/spaces/"
    if not value.startswith(marker):
        return value

    repo_id = value[len(marker) :]
    response = requests.get(f"{HF_ENDPOINT}/api/spaces/{repo_id}", timeout=30)
    response.raise_for_status()
    host = response.json().get("host")
    if host:
        return host.rstrip("/")
    return value


def build_remote_command(query: str) -> list[str]:
    return ["python", "run_collab_long.py", query]


def build_job_environment(
    cli_report_bucket: Optional[str],
    cli_report_prefix: Optional[str],
    cli_results_space_url: Optional[str],
) -> Dict[str, str]:
    env = {"PYTHONUNBUFFERED": "1", **collect_existing_env(DEFAULT_ENV_NAMES)}
    if cli_report_bucket:
        env["REPORT_BUCKET"] = cli_report_bucket
    if cli_report_prefix:
        env["REPORT_PREFIX"] = cli_report_prefix
    if cli_results_space_url:
        env["RESULTS_SPACE_URL"] = cli_results_space_url
    return env


def resolve_report_prefix(report_bucket: Optional[str], cli_report_prefix: Optional[str]) -> Optional[str]:
    if not report_bucket:
        return cli_report_prefix
    if cli_report_prefix:
        return normalize_report_prefix(cli_report_prefix, ".")
    return datetime.now().strftime("runs/%Y%m%d-%H%M%S")


def build_result_links(
    report_bucket: Optional[str],
    report_prefix: Optional[str],
    results_space_url: Optional[str],
    job_url: Optional[str],
) -> Optional[Dict[str, str]]:
    if not report_bucket or not report_prefix:
        return None
    normalized_prefix = normalize_report_prefix(report_prefix, ".")
    links = {
        "folder_url": bucket_folder_url(report_bucket, normalized_prefix),
        "report_view_url": bucket_file_view_url(report_bucket, normalized_prefix, "report.md"),
        "report_raw_url": bucket_file_resolve_url(report_bucket, normalized_prefix, "report.md"),
        "report_download_url": bucket_file_resolve_url(report_bucket, normalized_prefix, "report.md", download=True),
    }
    results_page_url = build_results_page_url(results_space_url, report_bucket, normalized_prefix)
    if results_page_url:
        links["results_page_url"] = results_page_url
    if job_url:
        links["job_url"] = job_url
    return links


def print_result_links(links: Optional[Dict[str, str]]):
    if not links:
        return
    print("\nRESULT LINKS")
    print("=" * 12)
    if links.get("results_page_url"):
        print(f"Open Results: {links['results_page_url']}")
    if links.get("job_url"):
        print(f"Job: {links['job_url']}")
    print(f"Report: {links['report_view_url']}")
    print(f"Download: {links['report_download_url']}")
    print(f"Folder: {links['folder_url']}")


def get_hf_token() -> Optional[str]:
    for name in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_timeout_seconds(timeout: str) -> int:
    factors = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if timeout and timeout[-1] in factors:
        return int(float(timeout[:-1]) * factors[timeout[-1]])
    return int(float(timeout))


def _build_rest_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _resolve_namespace_via_rest(token: str, namespace: Optional[str]) -> str:
    if namespace:
        return namespace
    response = requests.get(f"{HF_ENDPOINT}/api/whoami-v2", headers=_build_rest_headers(token), timeout=30)
    response.raise_for_status()
    return response.json()["name"]


def _job_web_url(job_id: str) -> str:
    return f"{HF_ENDPOINT}/jobs/{job_id}"


def should_fallback_to_rest(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, TypeError) and "JobOwner" in message


def submit_job_via_rest(
    *,
    image: str,
    command: list[str],
    env: Dict[str, str],
    secrets: Optional[Dict[str, str]],
    flavor: str,
    timeout: str,
    namespace: Optional[str],
):
    token = get_hf_token()
    if not token:
        raise RuntimeError("A Hugging Face token is required for REST fallback. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")

    resolved_namespace = _resolve_namespace_via_rest(token, namespace)
    payload = {
        "command": command,
        "arguments": [],
        "environment": env,
        "flavor": flavor,
        "timeoutSeconds": parse_timeout_seconds(timeout),
        "dockerImage": image,
    }
    if secrets:
        payload["secrets"] = secrets

    response = requests.post(
        f"{HF_ENDPOINT}/api/jobs/{resolved_namespace}",
        headers=_build_rest_headers(token),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return SimpleNamespace(
        id=data["id"],
        url=data.get("url") or _job_web_url(data["id"]),
        flavor=data.get("flavor", flavor),
        namespace=resolved_namespace,
    )


def inspect_job_via_rest(job_id: str, namespace: Optional[str]):
    token = get_hf_token()
    if not token:
        raise RuntimeError("A Hugging Face token is required for REST fallback. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")

    resolved_namespace = _resolve_namespace_via_rest(token, namespace)
    response = requests.get(
        f"{HF_ENDPOINT}/api/jobs/{resolved_namespace}/{job_id}",
        headers=_build_rest_headers(token),
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return SimpleNamespace(
        id=data["id"],
        status=SimpleNamespace(stage=data["status"]["stage"]),
        namespace=resolved_namespace,
    )


def fetch_job_logs_via_rest(job_id: str, namespace: Optional[str]):
    token = get_hf_token()
    if not token:
        raise RuntimeError("A Hugging Face token is required for REST fallback. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")

    resolved_namespace = _resolve_namespace_via_rest(token, namespace)
    response = requests.get(
        f"{HF_ENDPOINT}/api/jobs/{resolved_namespace}/{job_id}/logs",
        headers=_build_rest_headers(token),
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    for raw in response.iter_lines(chunk_size=1):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if line.startswith("data: {"):
            data = json.loads(line[len("data: ") :])
            message = data.get("data", "")
            if message and not message.startswith("===== Job started"):
                yield message


def wait_for_job(job_id: str, namespace: Optional[str], poll_interval: int) -> str:
    while True:
        try:
            job = HfApi().inspect_job(job_id=job_id, namespace=namespace)
            log_lines = lambda: HfApi().fetch_job_logs(job_id=job_id, namespace=namespace)
        except Exception as exc:
            if not should_fallback_to_rest(exc):
                raise
            job = inspect_job_via_rest(job_id, namespace)
            log_lines = lambda: fetch_job_logs_via_rest(job_id, namespace)
        stage = str(job.status.stage)
        print(f"[HF JOB] {job.id} stage={stage}", flush=True)
        if stage in {"COMPLETED", "ERROR", "CANCELED", "DELETED"}:
            print("[HF JOB] Final logs:", flush=True)
            for line in log_lines():
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
    parser.add_argument("--report-bucket", help="Bucket to upload final artifacts to, for example username/hf-agent")
    parser.add_argument("--report-prefix", help="Optional prefix inside the target bucket, for example runs/my-job")
    parser.add_argument("--results-space-url", help="Optional public Space URL used to render a clean results page")
    parser.add_argument("--wait", action="store_true", help="Wait for the remote job to finish")
    parser.add_argument("--poll-interval", type=int, default=15, help="Seconds between status polls when waiting")
    return parser.parse_args()


def main():
    args = parse_args()
    image = resolve_job_image(args.image)
    results_space_url = resolve_results_space_url(args.results_space_url)
    if not image:
        raise SystemExit(
            f"Set --image or define {DEFAULT_IMAGE_ENV} to a prebuilt Docker image that contains this repo."
        )

    resolved_report_prefix = resolve_report_prefix(args.report_bucket, args.report_prefix)
    env = build_job_environment(args.report_bucket, resolved_report_prefix, results_space_url)
    secrets = collect_existing_env(DEFAULT_SECRET_NAMES)

    try:
        job = run_job(
            image=image,
            command=build_remote_command(args.query),
            env=env,
            secrets=secrets or None,
            flavor=args.flavor,
            timeout=args.timeout,
            namespace=args.namespace,
        )
    except Exception as exc:
        if not should_fallback_to_rest(exc):
            raise
        print(f"[HF JOB] Client submission failed, falling back to raw REST call: {exc}", flush=True)
        job = submit_job_via_rest(
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
    print_result_links(build_result_links(args.report_bucket, resolved_report_prefix, results_space_url, job.url))

    if not args.wait:
        return

    stage = wait_for_job(job.id, args.namespace, args.poll_interval)
    print_result_links(build_result_links(args.report_bucket, resolved_report_prefix, results_space_url, job.url))
    sys.exit(0 if stage == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
