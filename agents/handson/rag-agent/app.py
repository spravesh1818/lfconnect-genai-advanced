import streamlit as st
from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from embeddings.embeddings import get_embeddings
from vectorstore.vector_store import create_vectorstore, get_retriever
from agent.graph import build_graph
from config import TOP_K
from dotenv import load_dotenv

load_dotenv()

st.title("📜 Constitution RAG Agent")

@st.cache_resource
def setup():
    docs = load_documents()
    chunks = split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)

    retriever = get_retriever(vectorstore, TOP_K)
    graph = build_graph(retriever)

    return graph

graph = setup()

query = st.text_input("Ask a question")

if st.button("Submit"):
    result = graph.invoke({
        "question": query,
        "context": "",
        "answer": "",
        "route": ""
    })

    st.write("### Answer")
    st.write(result["answer"])
