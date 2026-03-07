from typing import List
from state import SourceItem


def search_news(topic: str) -> List[SourceItem]:
    """
    Placeholder search tool.
    Replace later with Tavily / SerpAPI / News API / custom search.
    """
    return [
        {
            "title": f"Sample source about {topic}",
            "url": "https://example.com/sample-source",
            "snippet": f"This is a placeholder source for {topic}.",
            "source_type": "news",
        }
    ]