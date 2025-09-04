import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing_extensions import List

# create FAISS vector store
def create_FAISS(embedding_model: HuggingFaceEmbeddings) -> FAISS:
    index = faiss.IndexFlatL2(len(embedding_model.embed_query(" ")))
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    return vector_store

# add documents to FAISS vector store
def add_documents(faiss_vector_store: FAISS, documents: List[Document]) -> List[str]:
    return faiss_vector_store.add_documents(documents=documents)

# generate FAISS retriever
def get_FAISS_retriever(faiss_vector_store: FAISS, k: int = 5) -> VectorStoreRetriever:
    return faiss_vector_store.as_retriever(search_kwargs={"k": k})
