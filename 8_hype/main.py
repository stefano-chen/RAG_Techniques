import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from typing import Union, Tuple

load_dotenv("./.env")

# Constant definition
DATA_PATH = Path("./data/Understanding_Climate_Change.pdf")

LANGUAGE_MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Hypothetical Prompt Embedding
def generate_hypothetical_prompt_embeddings(chunk_text: Union[str,Document]) -> Tuple[Union[str, Document], list[list[float]]]:
    """
    Uses the LLM to generate multiple hypothetical questions for a single chunk.
    These questions will be used as 'proxies' for the chunk during retrieval.

    Params:
    chunk_text (str): Text contents of the chunk

    Returns:
    chunk_text (str): Text contents of the chunk. This is done to make the multithreading easier
    hypothetical prompt embedding (List[float]): A list of embedding vectors generated from the questions
    """
    llm = init_chat_model(model=LANGUAGE_MODEL_NAME, model_provider="google_genai")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    question_gen_prompt = PromptTemplate.from_template(
        "Analyze the input text and generate essential questions that, when answered," \
        "capture the main points of the text. Each question should be one line," \
        "without numbering or prefixes. \n\n" \
        "Text:\n{chunk_text}\n"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()

    questions = question_chain.invoke({"chunk_text": chunk_text}).replace("\n\n", "\n").split("\n")

    return chunk_text, embedding_model.embed_documents(questions)


def prepare_vector_store(chunks: list[str] | list[Document]):
    """
    Create and populates a FAISS vector store from a list of text chunks.
    
    This function processes a list of text chunks in parallel generating
    hypothetical prompt embeddings for each chunk.
    The embeddings are stored in a FAISS index for efficient similarity search.

    Parameters:
    chunks (List[str]): A list of text chunks to be embedde and stored.

    Returns: 
    FAISS: A FAISS vector store containing the embedded text chunks.
    """

    vector_store = None

    with ThreadPoolExecutor() as pool:
        # Use threading to speed up generation of prompt embeddings
        futures = [pool.submit(generate_hypothetical_prompt_embeddings, c) for c in chunks]

        # Process embeddings as they complete
        for f in tqdm(as_completed(futures), total=len(chunks)):

            chunk, vectors = f.result() # Retrieve the processed chunk and its embeddings

            # Initialize the FAISS vector store on the first chunk
            if vector_store == None:
                vector_store = FAISS(
                    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
                    index=faiss.IndexFlatL2(len(vectors[0])),
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
            
            # Pair the chunk's content with each generated embedding vector.
            # Each chunk is inserted multiple times, once for each prompt vector
            chunks_with_embbedding_vectors = [(chunk.page_content, vec) for vec in vectors]

            # IMPORTANT: This allow us to store the text and embeddings together
            vector_store.add_embeddings(chunks_with_embbedding_vectors)
    return vector_store


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents=documents)

    vectorstore = prepare_vector_store(texts)

    return vectorstore

chunks_vector_store = encode_pdf(DATA_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)