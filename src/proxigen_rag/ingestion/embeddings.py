
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class NomicEmbeddings(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [f"search_document: {text}" for text in texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        text = f"search_query: {text}"

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embedding_model():
    return NomicEmbeddings()
