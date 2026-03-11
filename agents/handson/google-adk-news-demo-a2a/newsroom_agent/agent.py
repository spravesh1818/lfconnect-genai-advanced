from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import (
    AGENT_CARD_WELL_KNOWN_PATH,
    RemoteA2aAgent,
)
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="openai/gpt-4o-mini")

remote_researcher = RemoteA2aAgent(
    name="remote_researcher",
    description="Remote newsroom research specialist available over A2A.",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

root_agent = Agent(
    name="root_newsroom_agent",
    model=MODEL,
    instruction=(
        "You are the managing editor. "
        "For greetings or simple conversation, answer directly. "
        "For recent-news research, delegate to remote_researcher. "
        "When you receive research notes back, summarize them clearly for the user."
    ),
    sub_agents=[remote_researcher],
)