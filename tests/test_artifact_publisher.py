import json
from pathlib import Path

from artifact_publisher import (
    build_artifacts_info,
    build_results_page_url,
    bucket_file_resolve_url,
    bucket_file_view_url,
    bucket_folder_url,
    normalize_report_prefix,
    update_results_space_index,
    upload_workspace_to_bucket,
    write_delivery_manifest,
    write_artifacts_manifest,
)


def test_normalize_report_prefix_uses_workspace_name_when_missing(tmp_path):
    shared_dir = tmp_path / "shared-123"
    shared_dir.mkdir()

    assert normalize_report_prefix(None, str(shared_dir)) == "runs/shared-123"


def test_build_artifacts_info_points_primary_to_report_when_present(tmp_path):
    (tmp_path / "report.md").write_text("# report\n")
    (tmp_path / "run_summary.json").write_text("{}\n")

    info = build_artifacts_info(str(tmp_path), "user/hf-agent", "runs/demo", "https://huggingface.co/spaces/acme/results")

    assert info["bucket_id"] == "user/hf-agent"
    assert info["bucket_prefix"] == "runs/demo"
    assert info["folder_url"] == "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo"
    assert info["results_page_url"] == "https://huggingface.co/spaces/acme/results?bucket=user%2Fhf-agent&prefix=runs%2Fdemo"
    assert info["primary_artifact"]["path"] == "report.md"
    assert info["primary_artifact"]["remote_uri"] == "hf://buckets/user/hf-agent/runs/demo/report.md"
    assert info["primary_artifact"]["view_url"] == "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo/report.md"
    assert info["primary_artifact"]["download_url"] == "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md?download=true"


def test_bucket_web_urls_are_constructed_consistently():
    assert bucket_folder_url("user/hf-agent", "runs/demo") == "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo"
    assert bucket_file_view_url("user/hf-agent", "runs/demo", "report.md") == "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo/report.md"
    assert bucket_file_resolve_url("user/hf-agent", "runs/demo", "report.md") == "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md"
    assert bucket_file_resolve_url("user/hf-agent", "runs/demo", "report.md", download=True) == "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md?download=true"
    assert build_results_page_url("https://huggingface.co/spaces/acme/results", "user/hf-agent", "runs/demo") == "https://huggingface.co/spaces/acme/results?bucket=user%2Fhf-agent&prefix=runs%2Fdemo"


def test_write_artifacts_manifest_writes_expected_json(tmp_path):
    artifacts = {
        "bucket_id": "user/hf-agent",
        "bucket_prefix": "runs/demo",
        "bucket_uri": "hf://buckets/user/hf-agent/runs/demo",
        "bucket_url": "https://huggingface.co/buckets/user/hf-agent",
        "folder_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo",
        "results_page_url": "https://huggingface.co/spaces/acme/results?bucket=user%2Fhf-agent&prefix=runs%2Fdemo",
        "primary_artifact": {
            "path": "report.md",
            "remote_uri": "hf://buckets/user/hf-agent/runs/demo/report.md",
            "view_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo/report.md",
            "raw_url": "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md",
            "download_url": "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md?download=true",
        },
        "files": [],
    }

    manifest_path = write_artifacts_manifest(str(tmp_path), artifacts, status="uploaded")
    payload = json.loads((tmp_path / "artifacts_manifest.json").read_text())

    assert manifest_path.endswith("artifacts_manifest.json")
    assert payload["status"] == "uploaded"
    assert payload["primary_artifact"]["path"] == "report.md"
    assert payload["results_page_url"] == "https://huggingface.co/spaces/acme/results?bucket=user%2Fhf-agent&prefix=runs%2Fdemo"


def test_write_delivery_manifest_writes_human_facing_links(tmp_path):
    artifacts = {
        "bucket_id": "user/hf-agent",
        "bucket_prefix": "runs/demo",
        "bucket_url": "https://huggingface.co/buckets/user/hf-agent",
        "folder_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo",
        "results_page_url": "https://huggingface.co/spaces/acme/results?bucket=user%2Fhf-agent&prefix=runs%2Fdemo",
        "primary_artifact": {
            "path": "report.md",
            "view_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo/report.md",
            "raw_url": "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md",
            "download_url": "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md?download=true",
        },
    }

    manifest_path = write_delivery_manifest(
        str(tmp_path),
        artifacts,
        task="Pick a model",
        status="pass",
        qa_passed=True,
        job_url="https://huggingface.co/jobs/user/job-123",
    )
    payload = json.loads((tmp_path / "delivery.json").read_text())

    assert manifest_path.endswith("delivery.json")
    assert payload["title"] == "Pick a model"
    assert payload["report_view_url"].endswith("/report.md")
    assert payload["results_page_url"].startswith("https://huggingface.co/spaces/")
    assert payload["job_url"].endswith("/jobs/user/job-123")


def test_upload_workspace_to_bucket_uses_sync_bucket(monkeypatch, tmp_path):
    (tmp_path / "report.md").write_text("# report\n")
    calls = {}

    def fake_sync_bucket(source, destination):
        calls["source"] = source
        calls["destination"] = destination

    monkeypatch.setattr("huggingface_hub.sync_bucket", fake_sync_bucket, raising=False)

    info = upload_workspace_to_bucket(str(tmp_path), "user/hf-agent", "runs/demo")

    assert calls["source"] == str(tmp_path)
    assert calls["destination"] == "hf://buckets/user/hf-agent/runs/demo"
    assert info["primary_artifact"]["path"] == "report.md"


def test_update_results_space_index_upserts_report_entry(monkeypatch, tmp_path):
    uploaded = {}
    existing_reports_path = tmp_path / "reports.json"
    existing_reports_path.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "id": "user/hf-agent::runs/older",
                        "title": "Older report",
                        "created_at": "2026-03-26T00:00:00+00:00",
                        "bucket_id": "user/hf-agent",
                        "bucket_prefix": "runs/older",
                    }
                ]
            }
        )
    )

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            uploaded["payload"] = json.loads(Path(path_or_fileobj).read_text())
            uploaded["path_in_repo"] = path_in_repo
            uploaded["repo_id"] = repo_id
            uploaded["repo_type"] = repo_type
            uploaded["commit_message"] = commit_message

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(existing_reports_path), raising=False)

    update_results_space_index(
        "user/hf-agent-results",
        {
            "title": "Fresh report",
            "status": "pass",
            "qa_passed": True,
            "created_at": "2026-03-27T00:00:00+00:00",
            "bucket_id": "user/hf-agent",
            "bucket_prefix": "runs/newer",
            "folder_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/newer",
            "report_view_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/newer/report.md",
            "report_download_url": "https://huggingface.co/buckets/user/hf-agent/resolve/runs/newer/report.md?download=true",
            "job_url": "https://huggingface.co/jobs/user/job-123",
        },
    )

    assert uploaded["repo_id"] == "user/hf-agent-results"
    assert uploaded["repo_type"] == "space"
    assert uploaded["path_in_repo"] == "reports.json"
    assert uploaded["payload"]["reports"][0]["bucket_prefix"] == "runs/newer"
    assert uploaded["payload"]["reports"][1]["bucket_prefix"] == "runs/older"
