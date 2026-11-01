from langgraph.graph import END, START, StateGraph

from proxigen_rag.graph.state import RAGState

from proxigen_rag.graph.nodes.input_guardrail import input_guardrail
from proxigen_rag.graph.nodes.guardrail_fallback import guardrail_fallback
from proxigen_rag.graph.nodes.save_question import save_question
from proxigen_rag.graph.nodes.prepare_query import prepare_query
from proxigen_rag.graph.nodes.query_analyzer import query_analyzer
from proxigen_rag.graph.nodes.direct_answer import direct_answer
from proxigen_rag.graph.nodes.retrieval_node import retrieval_node
from proxigen_rag.graph.nodes.answer_with_evidence import answer_with_evidence
from proxigen_rag.graph.nodes.verify_grounding import verify_grounding
from proxigen_rag.graph.nodes.fallback_answer import fallback_answer
from proxigen_rag.graph.nodes.save_answer import save_answer
from proxigen_rag.graph.nodes.reranker import reranker
from proxigen_rag.graph.nodes.retry_query import retry_query


def route_after_guardrail(state: RAGState) -> str:
    """Continue only if the user input passes the guardrail."""
    if state.get("input_allowed", False):
        return "save_question"

    return "guardrail_fallback"


def route_after_preparation(state: RAGState) -> str:
    """Ask for clarification or continue to query analysis."""
    if state.get("clarification_needed", False):
        return "save_answer"

    return "query_analyzer"


def route_after_query_analysis(state: RAGState) -> str:
    """Choose a direct response or retrieval-augmented answering."""
    if state.get("query_route") == "direct":
        return "direct_answer"

    return "retrieval_node"


def route_after_evidence_check(state: RAGState) -> str:

    status = state.get("evidence_status")
    attempts = state.get("retrieval_attempt", 0)

    if status in {"sufficient", "partial"}:
        return "verify_grounding"

    if status == "insufficient" and attempts < 1:
        return "retry_query"

    return "fallback_answer"


def route_after_grounding(state: RAGState) -> str:
    """Reject answers that contain unsupported claims."""
    if state.get("grounding_status") == "grounded":
        return "save_answer"

    return "fallback_answer"


def build_graph(checkpointer=None):
    builder = StateGraph(RAGState)

    builder.add_node("input_guardrail", input_guardrail)
    builder.add_node("guardrail_fallback", guardrail_fallback)
    builder.add_node("save_question", save_question)
    builder.add_node("prepare_query", prepare_query)
    builder.add_node("query_analyzer", query_analyzer)
    builder.add_node("direct_answer", direct_answer)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("answer_with_evidence", answer_with_evidence)
    builder.add_node("verify_grounding", verify_grounding)
    builder.add_node("fallback_answer", fallback_answer)
    builder.add_node("save_answer", save_answer)
    builder.add_node("reranker", reranker)
    builder.add_node("retry_query", retry_query)

    # Entry and input validation
    builder.add_edge(START, "input_guardrail")

    builder.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {
            "save_question": "save_question",
            "guardrail_fallback": "guardrail_fallback",
        },
    )

    # Save original question, then prepare it
    builder.add_edge("save_question", "prepare_query")

    builder.add_conditional_edges(
        "prepare_query",
        route_after_preparation,
        {
            "save_answer": "save_answer",
            "query_analyzer": "query_analyzer",
        },
    )

    # Decide whether retrieval is needed
    builder.add_conditional_edges(
        "query_analyzer",
        route_after_query_analysis,
        {
            "direct_answer": "direct_answer",
            "retrieval_node": "retrieval_node",
        },
    )

    # Direct responses
    builder.add_edge("direct_answer", "save_answer")

    # RAG path
    builder.add_edge("retrieval_node", "reranker")
    builder.add_edge("reranker", "answer_with_evidence")

    builder.add_conditional_edges(
        "answer_with_evidence",
        route_after_evidence_check,
        {
            "verify_grounding": "verify_grounding",
            "retry_query": "retry_query",
            "fallback_answer": "fallback_answer",
        },
    )

    builder.add_edge("retry_query", "retrieval_node")

    builder.add_conditional_edges(
        "verify_grounding",
        route_after_grounding,
        {
            "save_answer": "save_answer",
            "fallback_answer": "fallback_answer",
        },
    )

    # All final responses converge here
    builder.add_edge("fallback_answer", "save_answer")
    builder.add_edge("guardrail_fallback", "save_answer")
    builder.add_edge("save_answer", END)

    return builder.compile(checkpointer=checkpointer)