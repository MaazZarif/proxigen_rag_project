import logging

from langchain_core.prompts import ChatPromptTemplate

from proxigen_rag.graph.formatting import format_history
from proxigen_rag.graph.state import RAGState
from proxigen_rag.services.llm import get_groq_llm

logger = logging.getLogger(__name__)


from pydantic import BaseModel, Field


class DirectAnswer(BaseModel):
    answer: str = Field(
        description="A concise response to the user's message."
    )

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You handle greetings and ordinary conversation for Proxigen's website support assistant. Be polite, natural and concise.

Never state or invent Proxigen product facts, prices, specifications, strengths, policies or procedures. Product-support questions must go through the retrieval workflow, not be answered from general knowledge.
""",
        ),
        (
            "human",
            """
Recent conversation:
{history}

Current message:
{question}
""",
        ),
    ]
)

structured_llm = get_groq_llm().with_structured_output(DirectAnswer)


async def direct_answer(state: RAGState) -> dict:
    question = (state.get("question") or "").strip()

    history = format_history(
        state.get("messages", [])
    )

    try:

        chain = PROMPT | structured_llm
        response : DirectAnswer = await chain.ainvoke(
            {
                "history": history,
                "question": question,
            }
        )

        answer = response.answer.strip()

        return {"answer": answer or "How can I help you with Proxigen?"}

    except Exception:
        logger.exception("Direct answer generation failed.")

        return {
            "answer": (
                "I can help with Proxigen product questions. "
                "What would you like to know?"
            )
        }
