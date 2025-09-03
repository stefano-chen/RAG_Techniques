from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
from typing_extensions import List
from langchain_core.documents import Document
from .models import get_embedding_model

# Create FAISS vector store
def create_FAISS_store(documents: List[Document]) -> FAISS:
    return FAISS.from_documents(documents=documents, embedding=get_embedding_model("sentence-transformers/all-mpnet-base-v2"))

def create_FAISS_retriever(vector_store: FAISS, k: int = 2) -> VectorStoreRetriever:
    return vector_store.as_retriever(search_kwargs={"k": k})