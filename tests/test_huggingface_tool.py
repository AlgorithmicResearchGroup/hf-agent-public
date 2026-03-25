import json

from agent.tools.huggingface import huggingface_tool
from agent.tools.web import web_tool


class _FakeModel:
    id = "org/model"
    pipeline_tag = "text-generation"
    downloads = 123
    likes = 7
    last_modified = "2026-03-20T00:00:00"
    tags = ["transformers", "attention"]


class _FakeApi:
    def list_models(self, search, limit, full):
        assert search == "fast attention"
        assert limit == 1
        assert full is False
        return [_FakeModel()]


def test_search_huggingface_models_formats_results(monkeypatch):
    monkeypatch.setattr(huggingface_tool, "_get_api", lambda: _FakeApi())

    result = huggingface_tool.search_huggingface(
        {"resource_type": "models", "query": "fast attention", "limit": 1}
    )

    payload = json.loads(result["stdout"])
    assert result["status"] == "success"
    assert payload[0]["id"] == "org/model"
    assert payload[0]["pipeline_tag"] == "text-generation"


def test_normalize_fetch_url_prefers_ar5iv_for_arxiv():
    assert web_tool._normalize_fetch_url("https://arxiv.org/abs/2407.08608") == "https://ar5iv.labs.arxiv.org/html/2407.08608"
    assert web_tool._normalize_fetch_url("https://arxiv.org/pdf/2407.08608.pdf") == "https://ar5iv.labs.arxiv.org/html/2407.08608"


def test_search_web_fails_cleanly_without_tavily(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = web_tool.search_web({"query": "fast attention"})

    assert result["status"] == "failure"
    assert "TAVILY_API_KEY" in result["stderr"]
