import os
from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()

# Short-term memory: thread-level checkpointing
checkpointer = MemorySaver()

# Long-term memory: cross-thread persistent store
store = InMemoryStore()


def make_backend(runtime):
    # Files outside /memories live only in thread state
    # Files under /memories persist in the store across threads
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime)
        }
    )


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

agent = create_deep_agent(
    model=model,
    checkpointer=checkpointer,
    store=store,
    backend=make_backend,
    system_prompt=(
        "You are a helpful assistant with memory.\n"
        "Use the current conversation context when relevant.\n"
        "When the user shares a stable preference or fact they may want remembered "
        "across future conversations, save it under /memories/ as a small text file.\n"
        "When answering questions about prior preferences, check /memories/ if useful.\n"
        "Be concise."
    ),
)