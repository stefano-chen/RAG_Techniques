from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Fetch Embedding Model from HuggingFace
def get_embedding_model(full_model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=full_model_name)

# Fetch Google Gemini
def get_llm(full_model_name: str) -> BaseChatModel:
    return init_chat_model(model=full_model_name, model_provider="google_genai")

