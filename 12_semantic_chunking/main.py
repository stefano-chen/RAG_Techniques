from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv("./.env")

path = "data/Understanding_Climate_Change.pdf"

def read_pdf_to_string(path):
    loader = PyPDFLoader(path)
    return loader.load()

content = read_pdf_to_string(path=path)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

text_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

docs = text_splitter.split_documents(content)

vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

test_query = "What is the main cause of climate change?"
retrieved_docs = chunks_query_retriever.invoke(input=test_query)
for doc in retrieved_docs:
    print(doc.page_content, "\n", "-"*50)