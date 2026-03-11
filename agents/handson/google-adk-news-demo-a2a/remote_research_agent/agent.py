from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a

MODEL = LiteLlm(model="openai/gpt-4o-mini")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def research_topic(topic: str, days: str = "month") -> str:
    """Research a topic and return concise source-backed notes.

    Args:
        topic: The news topic to research.
        days: Time range for news search. Allowed values:
            'day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'.
            Common aliases are normalized automatically.
    """
    valid_time_ranges = {
        "day": "day",
        "d": "day",
        "today": "day",
        "daily": "day",
        "week": "week",
        "w": "week",
        "this_week": "week",
        "last_week": "week",
        "weekly": "week",
        "month": "month",
        "m": "month",
        "this_month": "month",
        "last_month": "month",
        "recent": "month",
        "latest": "month",
        "monthly": "month",
        "year": "year",
        "y": "year",
        "this_year": "year",
        "last_year": "year",
        "yearly": "year",
    }

    normalized_days = valid_time_ranges.get(str(days).strip().lower(), "month")

    result = tavily_client.search(
        query=topic,
        topic="news",
        time_range=normalized_days,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
        include_raw_content=False,
    )

    answer = result.get("answer", "")
    items = result.get("results", [])

    if not items:
        return f"No recent results found for: {topic}"

    lines = [
        f"Topic: {topic}",
        f"Time range: {normalized_days}",
        f"Summary: {answer}",
        "",
        "Sources:",
    ]

    for i, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")
        published_date = item.get("published_date", "")

        lines.append(f"{i}. {title}")
        if published_date:
            lines.append(f"   Published: {published_date}")
        lines.append(f"   URL: {url}")
        lines.append(f"   Note: {content}")

    return "\n".join(lines)

root_agent = Agent(
    name="remote_research_agent",
    model=MODEL,
    instruction=(
        "You are a remote research specialist for a newsroom. "
        "Use the research_topic tool when the user asks for recent news research. "
        "Return concise, source-backed notes."
    ),
    tools=[research_topic],
)

a2a_app = to_a2a(root_agent, port=8001)