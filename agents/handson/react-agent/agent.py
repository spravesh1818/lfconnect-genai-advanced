from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from state import AgentState
from tools import TOOLS

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a ReAct-style assistant.\n"
        "- Decide when a tool is needed.\n"
        "- If needed, call the tool with correct arguments.\n"
        "- Wait for the tool result before answering.\n"
        "- Never invent tool outputs.\n"
        "- Keep the final answer concise."
    )
)

_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_model_with_tools = _model.bind_tools(TOOLS)

def call_model(state: AgentState) -> AgentState:
    msgs = state["messages"]
    if not msgs or getattr(msgs[0], "type", None) != "system":
        msgs = [SYSTEM_PROMPT] + msgs

    response = _model_with_tools.invoke(msgs)
    return {"messages": [response]}