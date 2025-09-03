from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel

# fetch embedding model
def fetch_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)

def fetch_llm(model_name: str) -> BaseChatModel:
    return init_chat_model(model=model_name, model_provider="google_genai")