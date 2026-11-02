import logging
import os
from time import perf_counter

from langchain_core.documents import Document
from qdrant_client import models

from proxigen_rag.graph.state import RAGState
from proxigen_rag.ingestion.vector_store import get_qdrant_client
from proxigen_rag.ingestion.embeddings import get_embedding_model
from proxigen_rag.ingestion.sparse_embeddings import get_sparse_embedding_model

logger = logging.getLogger(__name__)


async def retrieval_node(state: RAGState) -> dict:
    total_start = perf_counter()

    try:
        start = perf_counter()

        search_query = (
            state.get("standalone_query")
            or state.get("question", "")
        ).strip()

        if not search_query:
            raise ValueError("Search query is empty")

        logger.info(
            "RETRIEVAL | query preparation=%.4fs",
            perf_counter() - start,
        )

        start = perf_counter()
        client = get_qdrant_client()

        logger.info(
            "RETRIEVAL | Qdrant client initialization=%.4fs",
            perf_counter() - start,
        )

        start = perf_counter()

        dense_model = get_embedding_model()
        dense_vector = await dense_model.aembed_query(search_query)

        logger.info(
            "RETRIEVAL | dense embedding=%.4fs",
            perf_counter() - start,
        )

        start = perf_counter()

        sparse_model = get_sparse_embedding_model()
        sparse_vector = list(sparse_model.query_embed(search_query))[0]

        logger.info(
            "RETRIEVAL | sparse embedding=%.4fs",
            perf_counter() - start,
        )

        start = perf_counter()

        response = await client.query_points(
            collection_name=os.environ["QDRANT_COLLECTION_NAME"],
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=10,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist(),
                    ),
                    using="sparse",
                    limit=10,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=5,
            with_payload=True,
        )

        logger.info(
            "RETRIEVAL | Qdrant hybrid search=%.4fs",
            perf_counter() - start,
        )

        start = perf_counter()

        documents = []

        for point in response.points:
            payload = point.payload or {}

            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )

        logger.info(
            "RETRIEVAL | document conversion=%.4fs | results=%d",
            perf_counter() - start,
            len(documents),
        )

        logger.info(
            "RETRIEVAL | TOTAL=%.4fs",
            perf_counter() - total_start,
        )

        return {"retrieved_documents": documents}

    except ValueError as e:
        logger.exception("Invalid input for retrieval: %s", e)
        raise

    except Exception as e:
        logger.exception("Hybrid retrieval failed: %s", e)
        raise