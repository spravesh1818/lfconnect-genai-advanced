from langchain_community.document_loaders import PyMuPDFLoader
from config import PDF_PATH

def load_documents():
    loader = PyMuPDFLoader(PDF_PATH)
    return loader.load()
