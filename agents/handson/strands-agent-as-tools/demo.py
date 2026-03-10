import os
from tavily import TavilyClient
from strands import Agent, tool
from dotenv import load_dotenv
from strands.models.openai import OpenAIModel

load_dotenv()

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
TODAY = "2026-03-09"

research_model = OpenAIModel(
    client_args={"api_key": os.environ["OPENAI_API_KEY"]},
    model_id="gpt-4o-mini",
    params={
        "max_tokens": 1000,
        "temperature": 0.1,
    },
)

writer_model = OpenAIModel(
    client_args={"api_key": os.environ["OPENAI_API_KEY"]},
    model_id="gpt-4o-mini",
    params={
        "max_tokens": 1000,
        "temperature": 0.4,
    },
)

editor_model = OpenAIModel(
    client_args={"api_key": os.environ["OPENAI_API_KEY"]},
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.1,
    },
)

@tool
def search_news(query: str, days: str = "month") -> str:
    """Search for recent news. Use for current events, funding, launches, and trends."""
    enriched_query = f"{query} 2026 latest"

    result = tavily.search(
        query=enriched_query,
        topic="news",
        time_range=days,
        search_depth="advanced",
        max_results=8,
        include_raw_content=False,
    )

    items = result.get("results", [])
    if not items:
        return f"No recent news results found for: {query}"

    lines = []
    for i, item in enumerate(items, start=1):
        title = item.get("title", "No title")
        url = item.get("url", "No URL")
        content = item.get("content", "")
        published = item.get("published_date", "Unknown date")
        lines.append(
            f"{i}. {title}\n"
            f"Published: {published}\n"
            f"URL: {url}\n"
            f"Snippet: {content}\n"
        )

    return "\n".join(lines)

@tool
def search_web(query: str) -> str:
    """Search the web for evergreen background information."""
    result = tavily.search(
        query=query,
        topic="general",
        search_depth="advanced",
        max_results=5,
        include_raw_content=False,
    )

    items = result.get("results", [])
    if not items:
        return f"No results found for: {query}"

    lines = []
    for i, item in enumerate(items, start=1):
        title = item.get("title", "No title")
        url = item.get("url", "No URL")
        content = item.get("content", "")
        lines.append(f"{i}. {title}\nURL: {url}\nSnippet: {content}\n")

    return "\n".join(lines)

@tool
def fetch_article(url: str) -> str:
    """Fetch a specific article/page and return extracted content."""
    result = tavily.extract(urls=[url])
    results = result.get("results", [])
    if not results:
        return f"Could not fetch article: {url}"

    page = results[0]
    raw_content = page.get("raw_content") or page.get("content") or ""
    return raw_content[:4000]

research_agent = Agent(
    model=research_model,
    system_prompt=(
        f"You are a newsroom researcher. Today's date is {TODAY}. "
        "For current topics, always use search_news first. "
        "Interpret 'recent' as current/2026 unless the user explicitly asks for another timeframe. "
        "Do not anchor on older results if the task asks for recent trends. "
        "Use fetch_article only for the most relevant recent sources. "
        "Return concise notes with URLs and published dates."
    ),
    tools=[search_news, search_web, fetch_article]
)

writer_agent = Agent(
    model=writer_model,
    system_prompt=(
        "You are a news writer. "
        "Write a clear, neutral, engaging short article from research notes."
    )
)

fact_check_agent = Agent(
    model=research_model,
    system_prompt=(
        f"You are a fact-checker. Today's date is {TODAY}. "
        "For recent claims, prefer search_news. "
        "Flag stale evidence and unsupported claims clearly."
    ),
    tools=[search_news, search_web, fetch_article]
)

@tool
def research_tool(topic: str) -> str:
    return str(research_agent(f"Research this topic: {topic}"))

@tool
def writer_tool(notes: str) -> str:
    return str(writer_agent(f"Write a short article from these notes:\n\n{notes}"))

@tool
def fact_check_tool(draft: str) -> str:
    return str(fact_check_agent(f"Fact-check this draft:\n\n{draft}"))

editor_agent = Agent(
    model=editor_model,
    system_prompt=(
        f"You are the managing editor. Today's date is {TODAY}. "
        "For news tasks, call research_tool first, then writer_tool, then fact_check_tool. "
        "If the user asks for recent trends, ensure the research is based on current-year or recent-month evidence."
    ),
    tools=[research_tool, writer_tool, fact_check_tool]
)

print(editor_agent(
    "Create a short article on AI startup funding trends in the last month. "
    "Use only recent sources unless briefly contrasting with older years."
))