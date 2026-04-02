# main.py
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv

from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from embeddings.embeddings import get_embeddings
from vectorstore.vector_store import create_vectorstore, get_retriever
from agent.graph import build_graph
from config import TOP_K

load_dotenv()

app = FastAPI()


def setup_graph():
    """
    Build and return the LangGraph RAG graph.

    This is called once at import time so the heavy work
    (loading PDFs, building embeddings, FAISS index, etc.)
    is not repeated on every request.
    """
    docs = load_documents()
    chunks = split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)

    retriever = get_retriever(vectorstore, TOP_K)
    graph = build_graph(retriever)

    return graph


# "Cached" graph: built once per process and reused
graph = setup_graph()


@app.post("/ask")
async def ask(payload: dict):
    """
    Non-streaming endpoint.
    Accepts: { "question": "..." }
    Returns: { "answer": "...", "context": "...", "route": "..." }
    """
    question = payload["question"]

    # Initial state for your graph (matches Streamlit app)
    state = {
        "question": question,
        "context": "",
        "answer": "",
        "route": "",
    }

    # Run the graph synchronously
    result = graph.invoke(state)

    # result should contain at least "answer" and maybe "route"/"context"
    return {
        "answer": result.get("answer", ""),
        "context": result.get("context", ""),
        "route": result.get("route", ""),
    }


@app.get("/stream")
async def stream(question: str, request: Request):
    """
    SSE endpoint.
    Query: /stream?question=...
    Streams the final answer as SSE.
    """
    async def event_generator():
        # Build initial state just like in /ask
        state = {
            "question": question,
            "context": "",
            "answer": "",
            "route": "",
        }

        # Run the graph (blocking, but simple for teaching)
        result = graph.invoke(state)

        # If client disconnected while we were working, stop
        if await request.is_disconnected():
            return

        # Send the answer as a single SSE event
        answer = result.get("answer", "")
        yield {
            "event": "answer",
            "data": answer,
        }

        # Optional "end" event so the client knows we're done
        yield {
            "event": "end",
            "data": "DONE",
        }

    return EventSourceResponse(event_generator())