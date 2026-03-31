from langchain_openai import ChatOpenAI
from retriever.retriever import retrieve_docs

from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieve_node(state, retriever):
    context = retrieve_docs(state["question"], retriever)
    return {"context": context}

def rag_node(state):
    prompt = f"""
    Answer ONLY using the context.

    Context:
    {state['context']}

    Question:
    {state['question']}
    """

    answer = llm.invoke(state["question"]).content
    return {"answer": answer}

def general_node(state):
    answer = llm.invoke(state["question"]).content
    return {"answer": answer}
