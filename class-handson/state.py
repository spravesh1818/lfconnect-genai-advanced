from __future__ import annotations
from dataclasses import dataclass,field
from typing import List,Dict,Literal,Optional


from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage

@dataclass
class ResearchState:
    messages:List[BaseMessage] = field(default_factory=list)
    notes: List[Dict[str,str]] = field(default_factory=list)
    next:Optional[Literal["web","wiki","synthesize","done"]] = None
    question:str = ""