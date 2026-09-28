"""search_web(query) - web search.

Uses Tavily (https://tavily.com, free tier) when TAVILY_API_KEY is set; otherwise falls back
to DuckDuckGo's Instant Answer API, which needs no key but only answers encyclopedic
queries (definitions, people, places) rather than returning full search results.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.tools.base import Permission, Tool, ToolError

TAVILY_URL = "https://api.tavily.com/search"
DUCKDUCKGO_URL = "https://api.duckduckgo.com/"


def _tavily(client: httpx.Client, api_key: str, query: str, max_results: int) -> dict[str, Any]:
    response = client.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "max_results": max_results, "include_answer": True},
    )
    response.raise_for_status()
    data = response.json()
    return {
        "provider": "tavily",
        "query": query,
        "answer": data.get("answer"),
        "results": [
            {"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:500]}
            for r in data.get("results", [])[:max_results]
        ],
    }


def _duckduckgo(client: httpx.Client, query: str, max_results: int) -> dict[str, Any]:
    response = client.get(DUCKDUCKGO_URL, params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
    response.raise_for_status()
    data = response.json()

    results = []
    if data.get("AbstractText"):
        results.append({"title": data.get("Heading"), "url": data.get("AbstractURL"), "snippet": data["AbstractText"]})
    for topic in data.get("RelatedTopics", []):
        for item in topic.get("Topics", [topic]):  # some entries are nested groups
            if item.get("Text"):
                results.append({"title": item["Text"][:80], "url": item.get("FirstURL"), "snippet": item["Text"]})
    return {
        "provider": "duckduckgo_instant_answer",
        "query": query,
        "answer": data.get("Answer") or None,
        "results": results[:max_results],
        "note": None if results else "No instant answer. Set TAVILY_API_KEY for full web search.",
    }


def make_search_tool(tavily_api_key: str | None = None, http_timeout: float = 10.0) -> Tool:
    def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
        query = query.strip()
        try:
            with httpx.Client(timeout=http_timeout) as client:
                if tavily_api_key:
                    return _tavily(client, tavily_api_key, query, max_results)
                return _duckduckgo(client, query, max_results)
        except httpx.TimeoutException as exc:
            raise ToolError("Search service timed out.", "timeout") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            kind = "rate_limited" if status == 429 else "upstream_error"
            raise ToolError(f"Search service returned HTTP {status}.", kind) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"Search service unavailable: {exc}", "upstream_error") from exc

    return Tool(
        name="search_web",
        description=(
            "Search the web for up-to-date or factual information you don't reliably know "
            "(recent events, facts about people, places, organizations). Returns titles, URLs and "
            "snippets. Cite the URLs you rely on."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query.", "minLength": 1, "maxLength": 400},
                "max_results": {"type": "integer", "description": "Number of results (1-10, default 5).", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        function=search_web,
        title="Web Search",
        permission=Permission.READ,
    )
