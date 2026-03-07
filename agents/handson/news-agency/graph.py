from langgraph.graph import StateGraph, START, END

from state import NewsState
from agents.researcher import researcher_graph
from agents.writer import writer_graph
from agents.fact_checker import fact_checker_graph
from agents.editor import editor_graph
from agents.publisher import publisher_node


def researcher_node(state: NewsState) -> dict:
    child_input = {
        "topic": state["topic"],
        "angle": state.get("angle", ""),
        "max_iterations": 4,
        "iteration": 0,
        "sources": state.get("sources", []),
        "research_notes": state.get("research_notes", []),
        "open_questions": state.get("open_questions", []),
        "fetched_pages": {},
        "logs": [],
    }

    result = researcher_graph.invoke(child_input)

    return {
        "sources": result.get("sources", []),
        "research_notes": result.get("research_notes", []),
        "open_questions": result.get("open_questions", []),
        "logs": [
            "Parent graph invoked researcher subgraph.",
            *result.get("logs", []),
        ],
    }


def writer_node(state: NewsState) -> dict:
    child_input = {
        "topic": state["topic"],
        "angle": state.get("angle", ""),
        "sources": state.get("sources", []),
        "research_notes": state.get("research_notes", []),
        "open_questions": state.get("open_questions", []),
        "fact_check_issues": state.get("fact_check_issues", []),
        "fact_check_summary": state.get("fact_check_summary", ""),
        "article_draft": state.get("article_draft", ""),
        "revision_count": state.get("revision_count", 0),
        "logs": [],
    }

    result = writer_graph.invoke(child_input)

    return {
        "headline": result.get("headline", ""),
        "subheadline": result.get("subheadline", ""),
        "article_outline": result.get("article_outline", ""),
        "article_draft": result.get("article_draft", ""),
        "logs": [
            "Parent graph invoked writer subgraph.",
            *result.get("logs", []),
        ],
    }


def fact_checker_node(state: NewsState) -> dict:
    child_input = {
        "topic": state["topic"],
        "angle": state.get("angle", ""),
        "article_draft": state.get("article_draft", ""),
        "sources": state.get("sources", []),
        "research_notes": state.get("research_notes", []),
        "verified_claims": [],
        "fact_check_issues": [],
        "fetched_pages": {},
        "iteration": 0,
        "max_iterations": 4,
        "logs": [],
    }

    result = fact_checker_graph.invoke(child_input)

    return {
        "verified_claims": result.get("verified_claims", []),
        "fact_check_issues": result.get("fact_check_issues", []),
        "fact_check_summary": result.get("fact_check_summary", ""),
        "logs": [
            "Parent graph invoked fact-checker subgraph.",
            *result.get("logs", []),
        ],
    }


def review_router_node(state: NewsState) -> dict:
    issues = state.get("fact_check_issues", [])
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)

    if revision_count >= max_revisions:
        return {
            "route_decision": "editor",
            "logs": [f"Router sent flow to editor because max revisions ({max_revisions}) reached."],
        }

    if not issues:
        return {
            "route_decision": "editor",
            "logs": ["Router sent flow to editor because there are no fact-check issues."],
        }

    high_issues = [issue for issue in issues if issue.get("severity") == "high"]
    medium_issues = [issue for issue in issues if issue.get("severity") == "medium"]

    needs_more_research = any(
        "missing evidence" in issue.get("problem", "").lower()
        or "could not verify" in issue.get("problem", "").lower()
        or "insufficient source" in issue.get("problem", "").lower()
        for issue in issues
    )

    if high_issues and needs_more_research:
        return {
            "route_decision": "research",
            "revision_count": revision_count + 1,
            "logs": ["Router sent flow back to researcher due to severe evidence gaps."],
        }

    if high_issues or medium_issues:
        return {
            "route_decision": "writer",
            "revision_count": revision_count + 1,
            "logs": ["Router sent flow back to writer for revision."],
        }

    return {
        "route_decision": "editor",
        "logs": ["Router sent flow to editor because only low-severity issues were found."],
    }


def editor_node(state: NewsState) -> dict:
    child_input = {
        "topic": state["topic"],
        "angle": state.get("angle", ""),
        "headline": state.get("headline", ""),
        "subheadline": state.get("subheadline", ""),
        "article_draft": state.get("article_draft", ""),
        "research_notes": state.get("research_notes", []),
        "fact_check_summary": state.get("fact_check_summary", ""),
        "fact_check_issues": state.get("fact_check_issues", []),
        "logs": [],
    }

    result = editor_graph.invoke(child_input)

    return {
        "headline": result.get("headline", state.get("headline", "")),
        "subheadline": result.get("subheadline", state.get("subheadline", "")),
        "article_draft": result.get("article_draft", state.get("article_draft", "")),
        "editor_notes": result.get("editor_notes", []),
        "logs": [
            "Parent graph invoked editor subgraph.",
            *result.get("logs", []),
        ],
    }


def route_after_review(state: NewsState) -> str:
    return state.get("route_decision", "editor")


def build_graph():
    builder = StateGraph(NewsState)

    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("review_router", review_router_node)
    builder.add_node("editor", editor_node)
    builder.add_node("publisher", publisher_node)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "fact_checker")
    builder.add_edge("fact_checker", "review_router")

    builder.add_conditional_edges(
        "review_router",
        route_after_review,
        {
            "research": "researcher",
            "writer": "writer",
            "editor": "editor",
        },
    )

    builder.add_edge("editor", "publisher")
    builder.add_edge("publisher", END)

    return builder.compile()


graph = build_graph()