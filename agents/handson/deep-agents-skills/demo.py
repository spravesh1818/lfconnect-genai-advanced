from agent import agent

tests = [
    "Write a blog about recent elections in Nepal",
]

for prompt in tests:
    print("\n" + "=" * 80)
    print("USER:", prompt)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    print("AGENT:")
    print(result["messages"][-1].content)