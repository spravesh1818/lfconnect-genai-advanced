from state import NewsState


def planner_node(state: NewsState) -> dict:
    topic = state["topic"]

    questions = [
        f"What happened regarding {topic}?",
        f"When did it happen?",
        f"Who are the main people or organizations involved in {topic}?",
        f"Why does {topic} matter?",
        f"What remains uncertain about {topic}?",
    ]

    return {
        "current_agent": "planner",
        "research_questions": questions,
        "status": "planned",
        "logs": [f"Planner created {len(questions)} research questions for topic: {topic}"]
    }