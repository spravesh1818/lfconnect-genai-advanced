from agent import agent

prompt = """
Write a short article on:
"Can LangChain Deep Agents collaborate with each other?"

Requirements:
- Keep it beginner-friendly
- Explain the role of subagents
- Mention one practical use case
- Keep it under 500 words
"""

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    },
    config={"configurable": {"thread_id": "newsroom-demo-1"}},
)

print(result["messages"][-1].content)