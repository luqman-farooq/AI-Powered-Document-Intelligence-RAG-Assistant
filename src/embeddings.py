from langchain_ollama import OllamaEmbeddings
from src.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
