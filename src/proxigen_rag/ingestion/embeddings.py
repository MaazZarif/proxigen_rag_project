from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


class NomicEmbeddings(Embeddings):

    def __init__(self):
        self.model = TextEmbedding(model_name=MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [f"search_document: {text}" for text in texts]
        return [normalize(v) for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        text = f"search_query: {text}"
        return normalize(next(iter(self.model.embed([text]))))


@lru_cache(maxsize=1)
def get_embedding_model():
    return NomicEmbeddings()