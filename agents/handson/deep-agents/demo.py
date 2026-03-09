from agent import agent


def ask(thread_id: str, text: str):
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": text}
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result["messages"]
    last = messages[-1]
    print(f"\n[thread={thread_id}] USER: {text}")
    print(f"[thread={thread_id}] AGENT: {last.content}")


if __name__ == "__main__":
    # -----------------------------
    # DEMO 1: short-term memory
    # -----------------------------
    # same_thread = "demo-thread-1"

    # ask(same_thread, "Hi, my name is Pravesh.")
    # ask(same_thread, "What is my name?")

    # # -----------------------------
    # # DEMO 2: long-term memory
    # # -----------------------------
    thread_a = "demo-thread-a"
    thread_b = "demo-thread-b"

    ask(
        thread_a,
        "Please remember across future conversations that I prefer concise bullet-free answers. "
        "Save it to long-term memory."
    )

    ask(
        thread_b,
        "Do you remember any answer-style preference I asked you to keep?"
    )