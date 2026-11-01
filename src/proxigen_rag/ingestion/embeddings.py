from functools import lru_cache
import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

@lru_cache(maxsize=1)
def get_embedding_model():
    return OllamaEmbeddings(
        model=os.environ["OLLAMA_EMBEDDING_MODEL"],
        base_url=os.environ["OLLAMA_BASE_URL"],
    )