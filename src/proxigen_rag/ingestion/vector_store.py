
import os
import uuid
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from langchain_qdrant import QdrantVectorStore

from proxigen_rag.ingestion.embeddings import get_embedding_model

load_dotenv()


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Async client for collection initialization and document ingestion."""
    return AsyncQdrantClient(
        url=os.environ["QDRANT_URL"],
    )


@lru_cache(maxsize=1)
def get_sync_qdrant_client():
    """Sync client for LangChain vector-store retrieval."""
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
    )


async def initialize_collection():
    client = get_qdrant_client()
    collection_name = os.environ["QDRANT_COLLECTION_NAME"]

    if await client.collection_exists(collection_name):
        return

    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )


async def store_documents(documents, texts, dense_vectors, sparse_vectors):
    client = get_qdrant_client()
    collection_name = os.environ["QDRANT_COLLECTION_NAME"]

    points = []

    for document, text, dense, sparse in zip(
        documents,
        texts,
        dense_vectors,
        sparse_vectors,
        strict=True,
    ):
        chunk_id = document.metadata["chunk_id"]

        points.append(
            models.PointStruct(
                id=str(uuid.UUID(chunk_id[:32])),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=sparse.indices,
                        values=sparse.values,
                    ),
                },
                payload={
                    "page_content": text,
                    "metadata": document.metadata,
                },
            )
        )

    if points:
        await client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )


@lru_cache(maxsize=1)
def get_vector_store():
    return QdrantVectorStore(
        client=get_sync_qdrant_client(),
        collection_name=os.environ["QDRANT_COLLECTION_NAME"],
        embedding=get_embedding_model(),
        vector_name="dense",
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )
