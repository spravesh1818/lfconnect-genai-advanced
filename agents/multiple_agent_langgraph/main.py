from langchain_core.messages import HumanMessage
from src.graph import graph

if __name__ == "__main__":
    user_request = "Plan me a 5-day trip in March. Suggest a destination, flights, and hotels."

    init_state = {
        "messages": [HumanMessage(content=user_request)],
        "destination": None,
        "flights": None,
        "hotels": None,
        "next": None,
    }

    final_state = graph.invoke(init_state)

    # Print the last AI message (final answer)
    print("\n--- FINAL ANSWER ---\n")
    print(final_state["messages"][-1].content)