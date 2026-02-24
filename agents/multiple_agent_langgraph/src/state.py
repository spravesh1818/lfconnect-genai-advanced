from typing import TypedDict, List, Optional, Literal
from langchain_core.messages import BaseMessage


class TravelState(TypedDict):
    messages: List[BaseMessage]

    # shared “working memory”
    destination: Optional[str]
    flights: Optional[str]
    hotels: Optional[str]

    # supervisor routing decision
    next: Optional[Literal["destination", "flight", "hotel", "final"]]