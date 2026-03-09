from agent import agent

prompts = [
    "Explain what Deep Agents are.",
    "What is 12 times 9?"
]

for prompt in prompts:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    print("\nUSER:", prompt)
    print("AGENT:", result["messages"][-1].content)