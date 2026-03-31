from langgraph.graph import StateGraph
from agent.state import AgentState
from agent.router import route_question
from agent.nodes import retrieve_node, rag_node, general_node

def build_graph(retriever):
    builder = StateGraph(AgentState)

    builder.add_node("router", route_question)
    builder.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    builder.add_node("rag", rag_node)
    builder.add_node("general", general_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "retrieve": "retrieve",
            "general": "general"
        }
    )

    builder.add_edge("retrieve", "rag")

    builder.set_finish_point("rag")
    builder.set_finish_point("general")

    return builder.compile()
