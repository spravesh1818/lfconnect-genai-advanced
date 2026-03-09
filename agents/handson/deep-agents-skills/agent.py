from pathlib import Path
from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=str(BASE_DIR)),
    skills=["./skills/"],
    system_prompt=(
        "You are a helpful assistant. "
        "Use available skills when they are relevant to the user's request. "
        "Otherwise answer normally."
    ),
)