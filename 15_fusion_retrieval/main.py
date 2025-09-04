from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv("./.env")

path = "data/Understanding_Climate_Change.pdf"

def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents=documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(texts, embedding=embeddings)

    return vectorstore, texts

vectorstore , cleaned_text = encode_pdf_and_get_split_documents(path=path)

# Create a bm25 index for retrieving documents by keywords
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    BM25 (Best Matching 25) is a ranking function used in information retrieval.
    It's based on the probabilistic retrieval framework and is an improvement over TF-IDF.

    Args:
    documents (List[Document]): List of documents to index

    Returns:
    BM25Okapi: An index that can be used for BM25 scoring.
    """

    # Tokenize each document by splitting on whitespace
    # This is a simple approach and could be improved with more sophisticated tokenization
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

bm25 = create_bm25_index(cleaned_text)

def fusion_retrieval(vectorstore: FAISS, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
    vectorstore (VectorStore): The vectorstore containing the documents.
    bm25 (BM25Okapi): Pre-computed BM25 index.
    query (str): The query string.
    k (int): The number of documents to retrieve.
    alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
    List[Document]: The top k documents based on the combined scores.
    """

    epsilon = 1e-8

    # Step 1: Get all documents from the vectorstore
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # Step 2: Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Step 3: Perform vector search
    vector_result = vectorstore.similarity_search_with_score(query=query, k=len(all_docs))

    # Step 4: Normalize scores
    vector_scores = np.array([score for _, score in vector_result])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # Step 5: Combine score
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    # Step 6: Rank documents
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Step 7: Return top k documents
    return [all_docs[i] for i in sorted_indices[:k]]


# Query
query = "What are the impacts of climate change on the environment?"

# Perform fusion retrieval
top_docs = fusion_retrieval(vectorstore=vectorstore, bm25=bm25, query=query, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]

context = f"\n{'-'*50}\n".join(docs_content)

print(context)