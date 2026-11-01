from langchain_core.messages import AIMessage

from proxigen_rag.graph.state import RAGState


def save_answer(state: RAGState) -> dict:
    answer = (state.get("answer") or "").strip()

    if not answer:
        answer = (
            "I'm unable to provide an answer right now. "
            "Please try again."
        )

    return {
        "messages": [AIMessage(content=answer)]
    }