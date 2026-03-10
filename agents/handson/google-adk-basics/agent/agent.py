from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()


def get_current_time() -> str:
    """Return the current local time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


root_agent = Agent(
    name="openai_basic_agent",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    description="A basic ADK agent powered by OpenAI",
    instruction=(
        "You are a helpful assistant. "
        "Use the get_current_time tool when the user asks for the current time."
    ),
    tools=[get_current_time],
)