from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.documents import Document
from typing_extensions import List

# Load CSV file
def load_csv(path_to_csv: str | Path) -> List[Document]:
    csv_loader = CSVLoader(file_path=path_to_csv)
    return csv_loader.load()

# Split CSV file in chunks
def split_csv(documents: List[Document], *, chunk_size=200, chunk_overlap=20) -> List[Document]:
    csv_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return csv_splitter.split_documents(documents=documents)
