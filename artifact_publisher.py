import json
import os
from typing import Any, Dict, List, Optional


IGNORED_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


def normalize_report_prefix(prefix: Optional[str], shared_dir: str) -> str:
    if prefix:
        cleaned = prefix.strip().strip("/")
        if cleaned:
            return cleaned
    return f"runs/{os.path.basename(os.path.abspath(shared_dir))}"


def bucket_destination(bucket_id: str, prefix: str) -> str:
    cleaned_prefix = normalize_report_prefix(prefix, ".")
    return f"hf://buckets/{bucket_id}/{cleaned_prefix}"


def bucket_browse_url(bucket_id: str) -> str:
    return f"https://huggingface.co/buckets/{bucket_id}"


def collect_workspace_files(shared_dir: str) -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for filename in sorted(filenames):
            if filename.endswith(".pyc"):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, shared_dir)
            files.append(rel)
    return sorted(files)


def build_artifacts_info(shared_dir: str, bucket_id: str, prefix: Optional[str]) -> Dict[str, Any]:
    normalized_prefix = normalize_report_prefix(prefix, shared_dir)
    destination = bucket_destination(bucket_id, normalized_prefix)
    files = collect_workspace_files(shared_dir)
    entries = []
    for rel in files:
        local_path = os.path.join(shared_dir, rel)
        remote_uri = f"{destination}/{rel}"
        entries.append(
            {
                "path": rel,
                "local_path": local_path,
                "remote_uri": remote_uri,
                "size_bytes": os.path.getsize(local_path),
            }
        )

    primary_rel = "report.md" if "report.md" in files else "run_summary.json"
    return {
        "bucket_id": bucket_id,
        "bucket_prefix": normalized_prefix,
        "bucket_uri": destination,
        "bucket_url": bucket_browse_url(bucket_id),
        "primary_artifact": {
            "path": primary_rel,
            "remote_uri": f"{destination}/{primary_rel}",
        },
        "files": entries,
    }


def write_artifacts_manifest(shared_dir: str, artifacts: Dict[str, Any], status: str, upload_error: Optional[str] = None) -> str:
    manifest = {
        "status": status,
        "bucket_id": artifacts["bucket_id"],
        "bucket_prefix": artifacts["bucket_prefix"],
        "bucket_uri": artifacts["bucket_uri"],
        "bucket_url": artifacts["bucket_url"],
        "primary_artifact": artifacts["primary_artifact"],
        "files": artifacts["files"],
    }
    if upload_error:
        manifest["upload_error"] = upload_error

    manifest_path = os.path.join(shared_dir, "artifacts_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def upload_workspace_to_bucket(shared_dir: str, bucket_id: str, prefix: Optional[str]) -> Dict[str, Any]:
    from huggingface_hub import sync_bucket

    normalized_prefix = normalize_report_prefix(prefix, shared_dir)
    artifacts = build_artifacts_info(shared_dir, bucket_id, normalized_prefix)
    destination = artifacts["bucket_uri"]
    sync_bucket(str(shared_dir), destination)
    return artifacts
