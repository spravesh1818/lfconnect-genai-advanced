# pip install langgraph langchain langchain-openai

import re
from typing import TypedDict, List, Annotated, Dict, Callable

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# -----------------------
# 1) Tools (classic: name + function)
# -----------------------
def calculator(expression: str) -> str:
    """Very basic calculator. (For real apps: use a safe evaluator.)"""
    return str(eval(expression))

def wiki_stub(query: str) -> str:
    """Stub tool. Replace with a real wiki/search tool if you want."""
    return f"(wiki_stub) Summary-ish result for: {query}"

TOOLS= {
    "calculator": calculator,
    "wiki": wiki_stub,
}


# -----------------------
# 2) ReAct prompt (classic format)
# -----------------------
SYSTEM = SystemMessage(
    content=(
        "You are a classic ReAct agent.\n\n"
        "You MUST follow this format and loop as needed:\n"
        "Thought: think about what to do\n"
        "Action: tool_name[tool_input]\n"
        "Observation: result of the action\n"
        "... (repeat Thought/Action/Observation as needed)\n"
        "Final: your final answer\n\n"
        "Available tools:\n"
        "- calculator[expression]\n"
        "- wiki[query]\n\n"
        "Rules:\n"
        "- Only call ONE action per step.\n"
        "- If you have enough info, respond with Final.\n"
        "- Never invent tool outputs. Use Observation from tools.\n"
    )
)


# -----------------------
# 3) LangGraph state
# -----------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# -----------------------
# 4) Helpers: parse Action and decide next step
# -----------------------
ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\[(.*)\]\s*$", re.DOTALL)

def extract_action(ai_text: str):
    """
    Returns (tool_name, tool_input) if Action line found, else None.
    We expect the model to output something ending with:
      Action: tool[input]
    """
    m = ACTION_RE.search(ai_text.strip())
    if not m:
        return None
    tool_name = m.group(1).strip()
    tool_input = m.group(2).strip()
    return tool_name, tool_input

def has_final(ai_text: str) -> bool:
    return "Final:" in ai_text


# -----------------------
# 5) Nodes
# -----------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def react_think_node(state: AgentState):
    """LLM produces the next Thought/Action or Final."""
    msgs = [SYSTEM] + state["messages"]
    resp = llm.invoke(msgs)
    return {"messages": [resp]}

def react_tool_node(state: AgentState):
    """
    Parse last AI message Action, run tool, append Observation as AIMessage text.
    """
    last_ai = state["messages"][-1]
    if not isinstance(last_ai, AIMessage):
        return {"messages": [AIMessage(content="Observation: (error) Expected AIMessage.")]}

    parsed = extract_action(last_ai.content)
    if not parsed:
        # No action found, return an observation to steer the agent
        return {"messages": [AIMessage(content="Observation: (error) No valid Action found. Use Action: tool[input].")]}

    tool_name, tool_input = parsed

    if tool_name not in TOOLS:
        return {"messages": [AIMessage(content=f"Observation: (error) Unknown tool '{tool_name}'.")]}

    try:
        result = TOOLS[tool_name](tool_input)
    except Exception as e:
        result = f"(tool error) {type(e).__name__}: {e}"

    # Classic ReAct feeds tool result back as Observation:
    return {"messages": [AIMessage(content=f"Observation: {result}")]}

def route_after_think(state: AgentState) -> str:
    """If Final -> end, else if Action -> tools, else tools with error observation."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and has_final(last.content):
        return "final"
    # If there's an Action, go to tools; otherwise go to tools to add error observation
    return "tools"


# -----------------------
# 6) Build graph (classic loop)
# -----------------------
graph = StateGraph(AgentState)

graph.add_node("think", react_think_node)
graph.add_node("tools", react_tool_node)

graph.set_entry_point("think")

graph.add_conditional_edges(
    "think",
    route_after_think,
    {"tools": "tools", "final": END},
)

graph.add_edge("tools", "think")  # loop

app = graph.compile()


# -----------------------
# 7) Run
# -----------------------
initial = {
    "messages": [
        HumanMessage(content="Compute (12*3)+10 using the calculator tool.")
    ]
}

final_state = app.invoke(initial)

print("\n--- FULL TRACE ---")
for m in final_state["messages"]:
    role = m.__class__.__name__.replace("Message", "")
    print(f"{role}: {m.content}\n")

print("=== FINAL ANSWER ===")
print(final_state["messages"][-1].content)