
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from proxigen_rag.graph.formatting import format_history
from proxigen_rag.graph.state import RAGState
from proxigen_rag.services.llm import get_groq_llm

logger = logging.getLogger(__name__)


class QueryPreparation(BaseModel):
    standalone_query: str = Field(
        description=(
            "A clear, standalone version of the user's current "
            "question, preserving its original intent."
        )
    )

    needs_clarification: bool = Field(
        description=(
            "True only when essential context is missing "
            "and cannot be resolved from conversation history."
        )
    )
    clarification_question: str = Field(
        default="",
        description=(
            "A short question asking only for the missing "
            "information, or an empty string when unnecessary."
        ),
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You prepare user queries for Proxigen's company and product knowledge assistant.
Task: normalize and contextualize the query. Do NOT answer it.

Rules:
1. Preserve the user's intent (a price question stays a price question, not a general product question).
2. Fix obvious spelling and grammar errors without changing meaning.
3. Use history to resolve references ("it", "that one", "its price", "the previous product") and short follow-ups.
4. Make the standalone_query understandable without history whenever possible. It is a search-ready question, not an answer and not a list of assumed facts.
5. Keep product identifiers, model numbers, codes, numerical values, units and technical terms exactly. Never silently change CMT01 to CTM01 or vice versa.
6. Never invent a product, identifier, specification, price, feature, company fact or user requirement.
7. Treat earlier assistant answers as conversational context, not verified evidence about Proxigen.
8. For questions like "What products does Proxigen offer?", keep the catalog intent; do not narrow it to one product.
9. "proxigen?" with no usable context: read it as "What is Proxigen?" only if that is the most reasonable interpretation; if essential ambiguity remains, ask briefly.
10. Do not ask for clarification when intent is already clear (e.g. the catalog question above).
11. Unrelated but clear general questions: preserve them, do not force them into a product question.

Return only the required structured fields.
            """,
        ),
        (
            "human",
            """
Recent conversation:
{history}

Current user question:
{question}
            """,
        ),
    ]
)


async def prepare_query(state: RAGState) -> dict:
    question = (state.get("question") or "").strip()

    if not question:
        return {
            "standalone_query": "",
            "clarification_needed": True,
            "clarification_question": "What would you like to know?",
        }

    try:
        history = format_history(state.get("messages", []))
        chain = PROMPT | get_groq_llm().with_structured_output(
            QueryPreparation
        )

        result: QueryPreparation = await chain.ainvoke(
            {
                "history": history or "No previous conversation.",
                "question": question,
            }
        )

        standalone_query = result.standalone_query.strip()

        if not standalone_query:
            standalone_query = question

        if result.needs_clarification:
            clarification = (
                result.clarification_question.strip()
                or "Which product or topic are you asking about?"
            )
            return {
                "standalone_query": standalone_query,
                "clarification_needed": True,
                "clarification_question": clarification,
                "answer": clarification,
            }

        return {
            "standalone_query": standalone_query,
            "clarification_needed": False,
            "clarification_question": "",
        }

    except Exception:
        logger.exception("Query preparation failed")

        return {
            "standalone_query": question,
            "clarification_needed": False,
            "clarification_question": "",
        }
