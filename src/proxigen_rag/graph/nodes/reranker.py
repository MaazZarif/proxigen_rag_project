import asyncio
import logging
from time import perf_counter

from sentence_transformers import CrossEncoder

from proxigen_rag.graph.state import RAGState

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5

reranker_model = CrossEncoder(
    MODEL_NAME,
    max_length=512,
    device="cpu",
)


async def reranker(state: RAGState) -> dict:
    start_time = perf_counter()

    question = (
    state.get("standalone_query")
    or state.get("question", "")
)

    documents = state.get("retrieved_documents", [])

    if not documents:
        logger.warning("RERANKING | No documents received")
        return {"retrieved_documents": []}

    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    scores = await asyncio.to_thread(
        reranker_model.predict,
        pairs,
    )

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    top_documents = [
        document
        for document, _ in ranked_documents[:TOP_K]
    ]

    logger.info(
        "RERANKING | candidates=%d | selected=%d | latency=%.3fs",
        len(documents),
        len(top_documents),
        perf_counter() - start_time,
    )

    return {"retrieved_documents": top_documents}