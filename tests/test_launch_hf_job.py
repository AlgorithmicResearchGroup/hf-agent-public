from launch_hf_job import (
    build_remote_command,
    get_hf_token,
    parse_timeout_seconds,
    resolve_job_image,
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
