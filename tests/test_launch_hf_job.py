from launch_hf_job import build_remote_command, resolve_job_image


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
