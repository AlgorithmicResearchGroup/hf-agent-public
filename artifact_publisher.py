import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode


IGNORED_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
HF_WEB_ENDPOINT = "https://huggingface.co"


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
    return f"{HF_WEB_ENDPOINT}/buckets/{bucket_id}"


def _quote_path(path: str) -> str:
    return quote(path.strip("/"), safe="/")


def bucket_folder_url(bucket_id: str, prefix: str) -> str:
    return f"{bucket_browse_url(bucket_id)}/tree/{_quote_path(normalize_report_prefix(prefix, '.'))}"


def bucket_file_view_url(bucket_id: str, prefix: str, relative_path: str) -> str:
    return f"{bucket_folder_url(bucket_id, prefix)}/{_quote_path(relative_path)}"


def bucket_file_resolve_url(bucket_id: str, prefix: str, relative_path: str, *, download: bool = False) -> str:
    url = f"{bucket_browse_url(bucket_id)}/resolve/{_quote_path(normalize_report_prefix(prefix, '.'))}/{_quote_path(relative_path)}"
    if download:
        return f"{url}?download=true"
    return url


def build_results_page_url(results_space_url: Optional[str], bucket_id: str, prefix: str) -> Optional[str]:
    if not results_space_url:
        return None
    base = results_space_url.rstrip("/")
    query = urlencode({"bucket": bucket_id, "prefix": normalize_report_prefix(prefix, ".")})
    return f"{base}?{query}"


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


def build_artifacts_info(
    shared_dir: str,
    bucket_id: str,
    prefix: Optional[str],
    results_space_url: Optional[str] = None,
) -> Dict[str, Any]:
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
                "view_url": bucket_file_view_url(bucket_id, normalized_prefix, rel),
                "raw_url": bucket_file_resolve_url(bucket_id, normalized_prefix, rel),
                "download_url": bucket_file_resolve_url(bucket_id, normalized_prefix, rel, download=True),
                "size_bytes": os.path.getsize(local_path),
            }
        )

    primary_rel = "report.md" if "report.md" in files else "run_summary.json"
    info = {
        "bucket_id": bucket_id,
        "bucket_prefix": normalized_prefix,
        "bucket_uri": destination,
        "bucket_url": bucket_browse_url(bucket_id),
        "folder_url": bucket_folder_url(bucket_id, normalized_prefix),
        "primary_artifact": {
            "path": primary_rel,
            "remote_uri": f"{destination}/{primary_rel}",
            "view_url": bucket_file_view_url(bucket_id, normalized_prefix, primary_rel),
            "raw_url": bucket_file_resolve_url(bucket_id, normalized_prefix, primary_rel),
            "download_url": bucket_file_resolve_url(bucket_id, normalized_prefix, primary_rel, download=True),
        },
        "files": entries,
    }
    results_page_url = build_results_page_url(results_space_url, bucket_id, normalized_prefix)
    if results_page_url:
        info["results_page_url"] = results_page_url
    return info


def write_artifacts_manifest(shared_dir: str, artifacts: Dict[str, Any], status: str, upload_error: Optional[str] = None) -> str:
    manifest = {
        "status": status,
        "bucket_id": artifacts["bucket_id"],
        "bucket_prefix": artifacts["bucket_prefix"],
        "bucket_uri": artifacts["bucket_uri"],
        "bucket_url": artifacts["bucket_url"],
        "folder_url": artifacts.get("folder_url"),
        "primary_artifact": artifacts["primary_artifact"],
        "files": artifacts["files"],
    }
    if artifacts.get("results_page_url"):
        manifest["results_page_url"] = artifacts["results_page_url"]
    if upload_error:
        manifest["upload_error"] = upload_error

    manifest_path = os.path.join(shared_dir, "artifacts_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def write_delivery_manifest(
    shared_dir: str,
    artifacts: Dict[str, Any],
    *,
    task: str,
    status: str,
    qa_passed: bool,
    job_url: Optional[str] = None,
) -> str:
    delivery = {
        "title": task,
        "status": status,
        "qa_passed": qa_passed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket_id": artifacts["bucket_id"],
        "bucket_prefix": artifacts["bucket_prefix"],
        "bucket_url": artifacts["bucket_url"],
        "folder_url": artifacts.get("folder_url"),
        "report_path": artifacts["primary_artifact"]["path"],
        "report_view_url": artifacts["primary_artifact"].get("view_url"),
        "report_raw_url": artifacts["primary_artifact"].get("raw_url"),
        "report_download_url": artifacts["primary_artifact"].get("download_url"),
    }
    if job_url:
        delivery["job_url"] = job_url
    if artifacts.get("results_page_url"):
        delivery["results_page_url"] = artifacts["results_page_url"]

    delivery_path = os.path.join(shared_dir, "delivery.json")
    with open(delivery_path, "w") as handle:
        json.dump(delivery, handle, indent=2)
    return delivery_path


def get_hf_token_from_env() -> Optional[str]:
    for name in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def update_results_space_index(
    results_space_repo_id: str,
    delivery: Dict[str, Any],
) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    token = get_hf_token_from_env()
    if not token:
        raise RuntimeError("Cannot update results Space index without HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory() as tmpdir:
        reports_path = Path(tmpdir) / "reports.json"
        try:
            existing_path = hf_hub_download(
                repo_id=results_space_repo_id,
                filename="reports.json",
                repo_type="space",
                token=token,
            )
            reports = json.loads(Path(existing_path).read_text()).get("reports", [])
        except EntryNotFoundError:
            reports = []

        entry_id = f"{delivery['bucket_id']}::{delivery['bucket_prefix']}"
        entry = {
            "id": entry_id,
            "title": delivery["title"],
            "status": delivery["status"],
            "qa_passed": delivery["qa_passed"],
            "created_at": delivery["created_at"],
            "bucket_id": delivery["bucket_id"],
            "bucket_prefix": delivery["bucket_prefix"],
            "folder_url": delivery["folder_url"],
            "report_view_url": delivery["report_view_url"],
            "report_download_url": delivery["report_download_url"],
            "job_url": delivery.get("job_url"),
        }

        reports = [report for report in reports if report.get("id") != entry_id]
        reports.append(entry)
        reports.sort(key=lambda report: report.get("created_at", ""), reverse=True)
        reports_path.write_text(json.dumps({"reports": reports}, indent=2), encoding="utf-8")

        api.upload_file(
            path_or_fileobj=str(reports_path),
            path_in_repo="reports.json",
            repo_id=results_space_repo_id,
            repo_type="space",
            commit_message=f"Update reports index for {delivery['bucket_prefix']}",
        )


def upload_workspace_to_bucket(
    shared_dir: str,
    bucket_id: str,
    prefix: Optional[str],
    results_space_url: Optional[str] = None,
) -> Dict[str, Any]:
    from huggingface_hub import sync_bucket

    normalized_prefix = normalize_report_prefix(prefix, shared_dir)
    artifacts = build_artifacts_info(shared_dir, bucket_id, normalized_prefix, results_space_url=results_space_url)
    destination = artifacts["bucket_uri"]
    sync_bucket(str(shared_dir), destination)
    return artifacts
