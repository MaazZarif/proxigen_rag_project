
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):


     # Input guardrail
    input_allowed: bool
    input_block_reason: str
    # Original user input and prepared search query and Clarification


    question: str
    standalone_query: str
    clarification_needed: bool
    clarification_question: str

    # Conversation history
    messages: Annotated[list[BaseMessage], add_messages]

   

    # Query analysis and routing
    query_route: Optional[Literal["direct", "rag"]]

    

    # Retrieval
    retrieved_documents: list[Any]

    # Evidence assessment and answer generation
    evidence_status: Optional[Literal["sufficient", "partial", "insufficient"]]
    evidence_reason: str
    answer: str

    # Grounding verification
    grounding_status: Optional[Literal["grounded", "ungrounded"]]
    grounding_reason: str
    unsupported_claims: list[str]

    retrieval_attempt: int
