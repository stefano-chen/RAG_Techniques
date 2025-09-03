from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing_extensions import List
from pathlib import Path


# Load pdf
def load_pdf(path_to_dir: str | Path) -> List[Document]:
    loader = PyPDFDirectoryLoader(path=path_to_dir)
    return loader.load()

# Split document in chunks
def split_pdf(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chucks = text_splitter.split_documents(documents=documents)
    return chucks
