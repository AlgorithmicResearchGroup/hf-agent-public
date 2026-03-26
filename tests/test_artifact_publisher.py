import json

from artifact_publisher import (
    build_artifacts_info,
    normalize_report_prefix,
    upload_workspace_to_bucket,
    write_artifacts_manifest,
)


def test_normalize_report_prefix_uses_workspace_name_when_missing(tmp_path):
    shared_dir = tmp_path / "shared-123"
    shared_dir.mkdir()

    assert normalize_report_prefix(None, str(shared_dir)) == "runs/shared-123"


def test_build_artifacts_info_points_primary_to_report_when_present(tmp_path):
    (tmp_path / "report.md").write_text("# report\n")
    (tmp_path / "run_summary.json").write_text("{}\n")

    info = build_artifacts_info(str(tmp_path), "user/hf-agent", "runs/demo")

    assert info["bucket_id"] == "user/hf-agent"
    assert info["bucket_prefix"] == "runs/demo"
    assert info["primary_artifact"]["path"] == "report.md"
    assert info["primary_artifact"]["remote_uri"] == "hf://buckets/user/hf-agent/runs/demo/report.md"


def test_write_artifacts_manifest_writes_expected_json(tmp_path):
    artifacts = {
        "bucket_id": "user/hf-agent",
        "bucket_prefix": "runs/demo",
        "bucket_uri": "hf://buckets/user/hf-agent/runs/demo",
        "bucket_url": "https://huggingface.co/buckets/user/hf-agent",
        "primary_artifact": {
            "path": "report.md",
            "remote_uri": "hf://buckets/user/hf-agent/runs/demo/report.md",
        },
        "files": [],
    }

    manifest_path = write_artifacts_manifest(str(tmp_path), artifacts, status="uploaded")
    payload = json.loads((tmp_path / "artifacts_manifest.json").read_text())

    assert manifest_path.endswith("artifacts_manifest.json")
    assert payload["status"] == "uploaded"
    assert payload["primary_artifact"]["path"] == "report.md"


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
