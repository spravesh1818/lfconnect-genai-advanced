from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, RERANK_TOP_K

reranker = CrossEncoder(RERANKER_MODEL)

def rerank(query, docs):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:RERANK_TOP_K]]

def retrieve_docs(query, retriever):
    query = "query: " + query  # important for BGE
    docs = retriever.invoke(query) 
    docs = rerank(query, docs)

    context = "\n\n".join([doc.page_content for doc in docs])
    return context
