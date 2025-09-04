from dotenv import load_dotenv
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

load_dotenv("./.env")

path = "data/Understanding_Climate_Change.pdf"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

def encode_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents=documents)

    return FAISS.from_documents(chunks, embedding=embeddings)

vectorstore = encode_pdf(path)


class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score oof a document to a query.")

def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )

    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score # type: ignore
        try:
            score = float(score)
        except ValueError: 
            score = 0 # Default score if parsing fails
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]

# Example usage of the reranking function
query = "What are the impacts of climate change on biodiversity?"
initial_docs = vectorstore.similarity_search(query=query, k=15)
reranked_docs = rerank_documents(query=query, docs=initial_docs)

# print first 3 initial documents
print("Top initial documents:")
for i, doc in enumerate(initial_docs[:3]):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:200] + "...") # Print first 200 characters of each document

# Print results
print(f"Query: {query}\n")
print("Top reranked documents:")
for i, doc in enumerate(reranked_docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:200] + "...")