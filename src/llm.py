from langchain_ollama import ChatOllama
from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=500,
        system=(
            "You are Nexa AI, a general-purpose AI assistant running locally. "
            "Reply in English when the user writes English. Reply in natural "
            "Pakistani Roman Urdu when the user writes Roman Urdu. Use Latin "
            "script for Roman Urdu and never use Devanagari. Answer the exact "
            "question, be accurate, and avoid unrelated information. "
            "Do not invent a creator, organization, or country of origin."
        ),
    )
