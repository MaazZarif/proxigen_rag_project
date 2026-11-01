
from proxigen_rag.graph.state import RAGState


async def guardrail_fallback(state: RAGState) -> dict:
    return {
        "answer": (
            "I couldn't process that message. Please ask a clear "
            "question about Proxigen's products or services."
        )
    }
