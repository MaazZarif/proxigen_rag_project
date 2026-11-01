from langchain_core.messages import HumanMessage

from proxigen_rag.graph.state import RAGState


def save_question(state: RAGState) -> dict:
    question = (state.get("question") or "").strip()

    return {
        "messages": [HumanMessage(content=question)],
        "standalone_query": "",
        "query_route": None,
        "retrieved_documents": [],
        "clarification_needed": False,
        "clarification_question": "",
        "evidence_status": None,
        "evidence_reason": "",
        "grounding_status": None,
        "grounding_reason": "",
        "unsupported_claims": [],
        "answer": "",
        "input_block_reason": "",
        "retrieval_attempt": 0,
    }