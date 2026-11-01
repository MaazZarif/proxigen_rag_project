
from uuid import uuid4
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    question: str
    answer: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, app_request: Request):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    thread_id = request.thread_id or str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:
        result = await app_request.app.state.graph.ainvoke(
            {"question": question},
            config=config,
        )

        answer = result.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            logger.error("Graph returned an empty or invalid answer")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate an answer.",
            )

        return ChatResponse(
            thread_id=thread_id,
            question=question,
            answer=answer.strip(),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Failed to process chat question")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from exc
