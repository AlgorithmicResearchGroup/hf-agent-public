import json
from pathlib import Path

from huggingface_hub import HfApi


huggingface_tool_definitions = [
    {
        "name": "search_huggingface",
        "description": "Search Hugging Face models, datasets, spaces, or papers relevant to a model-selection, fine-tuning, or capability-analysis task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "resource_type": {
                    "type": "string",
                    "description": "One of: models, datasets, spaces, papers.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 5.",
                },
            },
            "required": ["resource_type", "query"],
        },
    },
    {
        "name": "inspect_huggingface_repo",
        "description": "Inspect a Hugging Face model, dataset, or space repository and summarize its metadata plus top-level files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_type": {
                    "type": "string",
                    "description": "One of: model, dataset, space.",
                },
                "repo_id": {
                    "type": "string",
                    "description": "Repository identifier such as 'Qwen/Qwen2.5-0.5B'.",
                },
                "revision": {
                    "type": "string",
                    "description": "Optional revision or branch name.",
                },
                "files_limit": {
                    "type": "integer",
                    "description": "Maximum number of files to show from the repo tree. Default: 25.",
                },
            },
            "required": ["repo_type", "repo_id"],
        },
    },
    {
        "name": "read_huggingface_file",
        "description": "Download and read a text file from a Hugging Face model, dataset, or space repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_type": {
                    "type": "string",
                    "description": "One of: model, dataset, space.",
                },
                "repo_id": {
                    "type": "string",
                    "description": "Repository identifier.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file inside the repository.",
                },
                "revision": {
                    "type": "string",
                    "description": "Optional revision or branch name.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of characters to return. Default: 20000.",
                },
            },
            "required": ["repo_type", "repo_id", "path"],
        },
    },
]


def _get_api():
    return HfApi()


def _repo_type_value(repo_type: str) -> str:
    mapping = {
        "model": "model",
        "dataset": "dataset",
        "space": "space",
    }
    if repo_type not in mapping:
        raise ValueError(f"Unsupported repo_type '{repo_type}'. Expected one of: model, dataset, space.")
    return mapping[repo_type]


def _format_model_result(info):
    tags = ", ".join((getattr(info, "tags", None) or [])[:6])
    return {
        "id": getattr(info, "id", None) or getattr(info, "modelId", ""),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "last_modified": getattr(info, "last_modified", None),
        "tags": tags,
    }


def _format_dataset_result(info):
    tags = ", ".join((getattr(info, "tags", None) or [])[:6])
    return {
        "id": getattr(info, "id", None) or getattr(info, "datasetId", ""),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "last_modified": getattr(info, "last_modified", None),
        "tags": tags,
    }


def _format_space_result(info):
    sdk = getattr(info, "sdk", None)
    tags = ", ".join((getattr(info, "tags", None) or [])[:6])
    return {
        "id": getattr(info, "id", None),
        "sdk": sdk,
        "likes": getattr(info, "likes", None),
        "last_modified": getattr(info, "last_modified", None),
        "tags": tags,
    }


def _format_paper_result(info):
    paper = getattr(info, "paper", None)
    title = getattr(paper, "title", None) or getattr(info, "title", None)
    summary = getattr(paper, "summary", None) or getattr(info, "summary", None)
    source = getattr(paper, "source", None)
    return {
        "id": getattr(info, "id", None),
        "title": title,
        "published_at": getattr(info, "published_at", None),
        "source": getattr(source, "name", None) if source else None,
        "url": getattr(info, "url", None),
        "summary": summary,
    }


def search_huggingface(arguments, work_dir=None):
    api = _get_api()
    resource_type = arguments["resource_type"]
    query = arguments["query"]
    limit = arguments.get("limit", 5)

    if resource_type == "models":
        results = [_format_model_result(item) for item in api.list_models(search=query, limit=limit, full=False)]
    elif resource_type == "datasets":
        results = [_format_dataset_result(item) for item in api.list_datasets(search=query, limit=limit, full=False)]
    elif resource_type == "spaces":
        results = [_format_space_result(item) for item in api.list_spaces(search=query, limit=limit, full=False)]
    elif resource_type == "papers":
        results = []
        for idx, item in enumerate(api.list_papers(query=query)):
            if idx >= limit:
                break
            results.append(_format_paper_result(item))
    else:
        return {
            "tool": "search_huggingface",
            "status": "failure",
            "stdout": "",
            "stderr": "resource_type must be one of: models, datasets, spaces, papers",
        }

    return {
        "tool": "search_huggingface",
        "status": "success",
        "stdout": json.dumps(results, indent=2, default=str),
        "stderr": "",
    }


def inspect_huggingface_repo(arguments, work_dir=None):
    api = _get_api()
    repo_type = _repo_type_value(arguments["repo_type"])
    repo_id = arguments["repo_id"]
    revision = arguments.get("revision")
    files_limit = arguments.get("files_limit", 25)

    if repo_type == "model":
        info = api.model_info(repo_id, revision=revision, files_metadata=False)
    elif repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=revision, files_metadata=False)
    else:
        info = api.space_info(repo_id, revision=revision, files_metadata=False)

    files = []
    for idx, item in enumerate(api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, revision=revision, recursive=False)):
        if idx >= files_limit:
            break
        files.append(getattr(item, "path", str(item)))

    payload = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "sha": getattr(info, "sha", None),
        "last_modified": getattr(info, "last_modified", None),
        "likes": getattr(info, "likes", None),
        "downloads": getattr(info, "downloads", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": getattr(info, "tags", None),
        "card_data": getattr(info, "card_data", None),
        "siblings": files,
    }
    return {
        "tool": "inspect_huggingface_repo",
        "status": "success",
        "stdout": json.dumps(payload, indent=2, default=str),
        "stderr": "",
    }


def read_huggingface_file(arguments, work_dir=None):
    api = _get_api()
    repo_type = _repo_type_value(arguments["repo_type"])
    repo_id = arguments["repo_id"]
    path = arguments["path"]
    revision = arguments.get("revision")
    max_chars = arguments.get("max_chars", 20000)

    local_path = api.hf_hub_download(repo_id=repo_id, filename=path, repo_type=repo_type, revision=revision)
    content = Path(local_path).read_text(errors="replace")
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... [truncated to {max_chars} chars]"

    return {
        "tool": "read_huggingface_file",
        "status": "success",
        "stdout": content,
        "stderr": "",
    }
