from launch_hf_job import (
    build_result_links,
    build_job_environment,
    build_remote_command,
    get_hf_token,
    print_result_links,
    parse_timeout_seconds,
    resolve_job_image,
    resolve_report_prefix,
    should_fallback_to_rest,
    submit_job_via_rest,
)


def test_resolve_job_image_uses_cli_first(monkeypatch):
    monkeypatch.setenv("HF_JOB_IMAGE", "registry/env-image:latest")

    assert resolve_job_image("registry/cli-image:latest") == "registry/cli-image:latest"


def test_resolve_job_image_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HF_JOB_IMAGE", "registry/env-image:latest")

    assert resolve_job_image(None) == "registry/env-image:latest"


def test_build_remote_command_runs_orchestrator_from_image():
    command = build_remote_command(
        "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
    )

    assert command[:2] == ["python", "run_collab_long.py"]
    assert "What model on Hugging Face should I use?" in command[2]


def test_build_job_environment_adds_report_bucket_and_prefix(monkeypatch):
    monkeypatch.delenv("REPORT_BUCKET", raising=False)
    monkeypatch.delenv("REPORT_PREFIX", raising=False)
    monkeypatch.delenv("RESULTS_SPACE_URL", raising=False)

    env = build_job_environment("user/hf-agent", "runs/demo", "https://huggingface.co/spaces/acme/results")

    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["REPORT_BUCKET"] == "user/hf-agent"
    assert env["REPORT_PREFIX"] == "runs/demo"
    assert env["RESULTS_SPACE_URL"] == "https://huggingface.co/spaces/acme/results"


def test_resolve_report_prefix_uses_cli_value_when_present():
    assert resolve_report_prefix("user/hf-agent", "runs/demo") == "runs/demo"


def test_resolve_report_prefix_generates_default_for_bucket():
    prefix = resolve_report_prefix("user/hf-agent", None)

    assert prefix.startswith("runs/")


def test_build_result_links_includes_space_and_download_urls():
    links = build_result_links(
        "user/hf-agent",
        "runs/demo",
        "https://huggingface.co/spaces/acme/results",
        "https://huggingface.co/jobs/user/job-123",
    )

    assert links["results_page_url"] == "https://huggingface.co/spaces/acme/results?bucket=user%2Fhf-agent&prefix=runs%2Fdemo"
    assert links["report_view_url"] == "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo/report.md"
    assert links["report_download_url"] == "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md?download=true"
    assert links["job_url"] == "https://huggingface.co/jobs/user/job-123"


def test_print_result_links_outputs_space_url_first(capsys):
    print_result_links(
        {
            "results_page_url": "https://huggingface.co/spaces/acme/results?bucket=user%2Fhf-agent&prefix=runs%2Fdemo",
            "job_url": "https://huggingface.co/jobs/user/job-123",
            "report_view_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo/report.md",
            "report_download_url": "https://huggingface.co/buckets/user/hf-agent/resolve/runs/demo/report.md?download=true",
            "folder_url": "https://huggingface.co/buckets/user/hf-agent/tree/runs/demo",
        }
    )

    output = capsys.readouterr().out.strip().splitlines()
    assert output[0] == "RESULT LINKS"
    assert output[2].startswith("Open Results: ")


def test_get_hf_token_prefers_hub_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-token")

    assert get_hf_token() == "hub-token"


def test_parse_timeout_seconds_handles_suffixes():
    assert parse_timeout_seconds("2h") == 7200
    assert parse_timeout_seconds("90m") == 5400
    assert parse_timeout_seconds("30") == 30


def test_should_fallback_to_rest_on_jobowner_parse_bug():
    exc = TypeError("JobOwner.__init__() got an unexpected keyword argument 'type'")

    assert should_fallback_to_rest(exc) is True


def test_submit_job_via_rest_posts_expected_payload(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-token")

    calls = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, headers=None, timeout=None):
        calls["get"] = {"url": url, "headers": headers, "timeout": timeout}
        return FakeResponse({"name": "test-user"})

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["post"] = {"url": url, "headers": headers, "json": json, "timeout": timeout}
        return FakeResponse({"id": "job-123", "flavor": "cpu-basic"})

    monkeypatch.setattr("launch_hf_job.requests.get", fake_get)
    monkeypatch.setattr("launch_hf_job.requests.post", fake_post)

    job = submit_job_via_rest(
        image="ghcr.io/org/repo:latest",
        command=["python", "run_collab_long.py", "query"],
        env={"MAX_ITERATIONS": "20"},
        secrets={"OPENAI": "secret"},
        flavor="cpu-basic",
        timeout="2h",
        namespace=None,
    )

    assert job.id == "job-123"
    assert job.url.endswith("/jobs/job-123")
    assert calls["post"]["url"].endswith("/api/jobs/test-user")
    assert calls["post"]["json"]["dockerImage"] == "ghcr.io/org/repo:latest"
    assert calls["post"]["json"]["timeoutSeconds"] == 7200
    assert calls["post"]["json"]["secrets"] == {"OPENAI": "secret"}
