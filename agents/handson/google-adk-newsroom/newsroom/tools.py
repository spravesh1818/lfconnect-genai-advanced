from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

from google.adk.tools.tool_context import ToolContext

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def tavily_news_search(
    query: str,
    time_range: str = "month",
    max_results: int = 5,
) -> str:
    """Search recent news using Tavily and return structured notes."""
    response = tavily_client.search(
        query=query,
        topic="news",
        time_range=time_range,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )

    results = response.get("results", [])
    answer = response.get("answer", "")

    if not results:
        return json.dumps(
            {
                "query": query,
                "answer": answer,
                "results": [],
                "note": "No results found.",
            },
            indent=2,
            ensure_ascii=False,
        )

    normalized_results = []
    for item in results:
        normalized_results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
                "published_date": item.get("published_date"),
            }
        )

    return json.dumps(
        {
            "query": query,
            "answer": answer,
            "results": normalized_results,
        },
        indent=2,
        ensure_ascii=False,
    )


def set_news_topic(topic: str, tool_context: ToolContext) -> str:
    tool_context.state["news_topic"] = topic
    tool_context.state["news_updated_at"] = datetime.utcnow().isoformat()
    return f"Stored topic: {topic}"


def get_news_topic(tool_context: ToolContext) -> str:
    topic = tool_context.state.get("news_topic")
    if not topic:
        return "No topic is stored yet."
    return f"Current topic: {topic}"


def save_article_metadata(
    headline: str,
    angle: str,
    keywords: list[str],
    tool_context: ToolContext,
) -> str:
    tool_context.state["article_metadata"] = {
        "headline": headline,
        "angle": angle,
        "keywords": keywords,
        "saved_at": datetime.utcnow().isoformat(),
    }
    return "Article metadata saved."


def get_article_context(tool_context: ToolContext) -> str:
    payload: dict[str, Any] = {
        "topic": tool_context.state.get("news_topic"),
        "research_notes": tool_context.state.get("research_notes"),
        "draft_article": tool_context.state.get("draft_article"),
        "final_article": tool_context.state.get("final_article"),
        "article_metadata": tool_context.state.get("article_metadata"),
        "updated_at": tool_context.state.get("news_updated_at"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def clear_current_story(tool_context: ToolContext) -> str:
    keys = [
        "news_topic",
        "research_notes",
        "draft_article",
        "final_article",
        "article_metadata",
        "news_updated_at",
    ]

    for key in keys:
        tool_context.state[key] = None

    return "Cleared the current newsroom story."