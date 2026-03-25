import os
import re

import requests
import trafilatura
from tavily import TavilyClient


web_tool_definitions = [
    {
        "name": "search_web",
        "description": "Search the web for current information using Tavily. Returns titles, URLs, and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return. Default: 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": "Fetch a URL and extract readable text content. Use this for papers, blog posts, docs, and reference pages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
]


def _normalize_fetch_url(url: str) -> str:
    if "ar5iv.labs.arxiv.org" in url:
        return url

    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)(?:v\d+)?(?:\.pdf)?", url)
    if match:
        return f"https://ar5iv.labs.arxiv.org/html/{match.group(1)}"

    return url


def search_web(arguments, work_dir=None):
    query = arguments["query"]
    num_results = arguments.get("num_results", 5)

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "tool": "search_web",
            "status": "failure",
            "stdout": "",
            "stderr": "TAVILY_API_KEY is not set. Use search_huggingface or fetch_page for direct-source work, or configure Tavily for broader web search.",
        }

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=num_results)

    results = response["results"]
    formatted_lines = []
    for i, r in enumerate(results, 1):
        formatted_lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['content'][:300]}")

    output = "\n\n".join(formatted_lines) if formatted_lines else "No results found."

    return {
        "tool": "search_web",
        "status": "success",
        "stdout": output,
        "stderr": "",
    }


def fetch_page(arguments, work_dir=None):
    original_url = arguments["url"]
    url = _normalize_fetch_url(original_url)

    headers = {"User-Agent": "multiagent-hf/1.0"}
    text = None
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        text = trafilatura.extract(response.text)
    except requests.RequestException as exc:
        return {
            "tool": "fetch_page",
            "status": "error",
            "stdout": "",
            "stderr": f"Failed to fetch {original_url}: {exc}",
        }

    if not text:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)

    if not text:
        if original_url.lower().endswith(".pdf") and "arxiv.org" not in original_url:
            error = (
                f"Could not extract text from PDF URL {original_url}. "
                "Prefer an HTML page, arXiv abstract page, or ar5iv mirror."
            )
        else:
            error = f"Could not extract text from {original_url}"
        return {
            "tool": "fetch_page",
            "status": "error",
            "stdout": "",
            "stderr": error,
        }

    return {
        "tool": "fetch_page",
        "status": "success",
        "stdout": text,
        "stderr": "",
    }
