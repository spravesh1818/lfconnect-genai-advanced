import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from tavily import TavilyClient

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SKILLS_DIR = BASE_DIR / "skills"

# Shared working area for the newsroom pipeline
WORKSPACE_DIR = BASE_DIR / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
):
    """Search the web for relevant information."""
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=False,
    )


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

researcher = {
    "name": "researcher",
    "description": "Researches a topic and writes structured notes to the shared workspace.",
    "system_prompt": (
        "You are the newsroom researcher.\n"
        "Your job is to research the user's requested topic using available tools.\n"
        "Write your final research notes to /workspace/research.md.\n\n"
        "Your research notes must include:\n"
        "1. Topic summary\n"
        "2. Key facts\n"
        "3. Important source URLs or source titles\n"
        "4. Open questions or uncertainties\n\n"
        "Be factual, structured, and non-stylistic.\n"
        "Do not write the article itself."
    ),
    "tools": [internet_search],
}

writer = {
    "name": "writer",
    "description": "Reads research notes and writes the first article draft.",
    "system_prompt": (
        "You are the newsroom writer.\n"
        "Read /workspace/research.md.\n"
        "Write a strong first draft to /workspace/draft.md.\n"
        "Do not invent facts.\n"
        "Use only the research provided.\n"
        "Do not ask for a file path; use the shared workspace paths directly."
    ),
    "skills": [str(SKILLS_DIR / "writer") + "/"],
}

editor = {
    "name": "editor",
    "description": "Reads the draft and improves clarity, flow, and readability.",
    "system_prompt": (
        "You are the newsroom editor.\n"
        "Read /workspace/draft.md.\n"
        "Revise it for clarity, structure, flow, brevity, and consistency.\n"
        "Write the improved version to /workspace/edited.md.\n"
        "Do not add unsupported factual claims.\n"
        "Do not ask for a file path; use the shared workspace paths directly."
    ),
    "skills": [str(SKILLS_DIR / "editor") + "/"],
}

publisher = {
    "name": "publisher",
    "description": "Reads the edited article and formats the final publication-ready version.",
    "system_prompt": (
        "You are the newsroom publisher.\n"
        "Read /workspace/edited.md.\n"
        "Prepare the final publication-ready article and write it to /workspace/final.md.\n"
        "Make it clean, well-formatted, and easy to read.\n"
        "Do not ask for a file path; use the shared workspace paths directly."
    ),
    "skills": [str(SKILLS_DIR / "publisher") + "/"],
}

agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(
        root_dir=str(BASE_DIR),
        virtual_mode=True,
    ),
    subagents=[researcher, writer, editor, publisher],
    system_prompt=(
        "You are the editor-in-chief of a small AI newsroom.\n"
        "Coordinate the available subagents to complete article requests.\n\n"
        "Always use the shared filesystem workflow below:\n"
        "1. Researcher writes research notes to /workspace/research.md\n"
        "2. Writer reads /workspace/research.md and writes /workspace/draft.md\n"
        "3. Editor reads /workspace/draft.md and writes /workspace/edited.md\n"
        "4. Publisher reads /workspace/edited.md and writes /workspace/final.md\n"
        "5. Read /workspace/final.md and return its contents to the user\n\n"
        "Delegate the work instead of doing every step yourself.\n"
        "Do not ask downstream agents to work on a document unless the upstream file exists.\n"
        "If a step fails, inspect the workspace files and recover."
    ),
)