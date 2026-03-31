from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

def get_retriever(vectorstore, k):
    return vectorstore.as_retriever(search_kwargs={"k": k})
