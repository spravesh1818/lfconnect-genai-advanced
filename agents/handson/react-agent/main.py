from dotenv import load_dotenv
load_dotenv()

from graph import graph

def print_last_message(messages):
    last = messages[-1]
    print("\n================= STEP =================")
    print(f"Type: {getattr(last, 'type', type(last))}")

    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("Tool calls requested:")
        for tc in tool_calls:
            print(f"  - {tc.get('name')}({tc.get('args')})")

    content = getattr(last, "content", None)
    if content:
        print("Content:")
        print(content)

def run():
    inputs = {
        "messages": [
            ("user", "What is 17 times 8, and what is the weather in Kathmandu?")
        ]
    }

    for event in graph.stream(inputs, stream_mode="values"):
        print_last_message(event["messages"])

    final_state = graph.invoke(inputs)
    print("\n================= FINAL ANSWER =================")
    print(final_state["messages"][-1].content)

if __name__ == "__main__":
    run()