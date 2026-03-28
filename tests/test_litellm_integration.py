import sys
from types import SimpleNamespace

import agent.utils as utils
import agent.worker as worker_module
from run_collab_long import call_assign_fixes_litellm, call_orchestrator_litellm


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, tool_calls=None, content=None, role="assistant"):
        self.tool_calls = tool_calls or []
        self.content = content
        self.role = role

    def model_dump(self):
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


def test_ensure_litellm_env_maps_existing_repo_env_vars(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI", "openai-secret")
    monkeypatch.setenv("ANTHROPIC", "anthropic-secret")

    utils.ensure_litellm_env()

    assert utils.os.environ["OPENAI_API_KEY"] == "openai-secret"
    assert utils.os.environ["ANTHROPIC_API_KEY"] == "anthropic-secret"


def test_worker_uses_litellm_model_for_litellm_provider(monkeypatch, tmp_path):
    captured = {}

    class DummyLiteLLMModel:
        def __init__(self, system_prompt, all_tools, model_name, max_tokens):
            captured["system_prompt"] = system_prompt
            captured["all_tools"] = all_tools
            captured["model_name"] = model_name
            captured["max_tokens"] = max_tokens

    monkeypatch.setattr(worker_module, "LiteLLMModel", DummyLiteLLMModel)
    monkeypatch.setattr(worker_module, "AgentMemory", lambda: object())
    monkeypatch.setattr(worker_module, "get_all_tools", lambda *args, **kwargs: [])
    monkeypatch.setattr(worker_module, "get_worker_system_prompt", lambda *args, **kwargs: "system")

    worker = worker_module.Worker(
        user_id=1,
        run_id=1,
        user_query="pick a model",
        worker_number=1,
        provider="litellm",
        model_name="openai/gpt-5.2",
        max_tokens=2048,
        work_dir=str(tmp_path),
    )

    assert isinstance(worker.model, DummyLiteLLMModel)
    assert captured["model_name"] == "openai/gpt-5.2"
    assert captured["max_tokens"] == 2048


def test_worker_cost_estimate_supports_litellm_prefixed_models():
    worker = worker_module.Worker.__new__(worker_module.Worker)
    worker.model_name = "openai/gpt-5.2"

    cost = worker._estimate_cost(prompt_tokens=1_000_000, response_tokens=1_000_000)

    assert cost == 12.5


def test_call_orchestrator_litellm_parses_tool_response(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            "call_1",
                            "decompose_task",
                            '{"agents":[{"id":"landscape","role":"HF researcher","task":"Write huggingface_ecosystem.md","depends_on":[]}]}',
                        )
                    ]
                )
            )
        ]
    )
    fake_module = SimpleNamespace(completion=lambda **kwargs: response)
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setenv("OPENAI", "openai-secret")

    result = call_orchestrator_litellm("openai/gpt-5.2", "Find the best coding model.")

    assert result["agents"][0]["id"] == "landscape"


def test_call_assign_fixes_litellm_accepts_dict_arguments(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=FakeMessage(
                    tool_calls=[
                        {
                            "id": "call_2",
                            "function": {
                                "name": "assign_fixes",
                                "arguments": {
                                    "assignments": [
                                        {
                                            "agent_id": "impl",
                                            "error_ids": ["err_1"],
                                            "fix_task": "Add missing snippet explanation.",
                                        }
                                    ]
                                },
                            },
                        }
                    ]
                )
            )
        ]
    )
    fake_module = SimpleNamespace(completion=lambda **kwargs: response)
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setenv("OPENAI", "openai-secret")

    report = {"errors": [{"id": "err_1", "severity": "critical", "file": "report.md", "description": "Missing snippet explanation", "evidence": "No line-by-line notes"}]}
    agents = [{"id": "impl", "role": "Implementation", "task": "Write code_examples.md"}]

    result = call_assign_fixes_litellm("openai/gpt-5.2", report, agents)

    assert result["assignments"][0]["agent_id"] == "impl"
