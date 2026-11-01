from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from proxigen_rag.services.llm import get_groq_llm
from proxigen_rag.graph.state import RAGState


async def retry_query(state: RAGState) -> dict:

    question = state.get("question", "")
    previous_query = state.get("standalone_query", question)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Rewrite the search query to improve document retrieval.

            Preserve important technical terms, product codes,
            and specifications.

            Do not answer the question.
            Return only the improved search query."""
        ),
        (
            "human",
            """Original question: {question}

            Previous search query: {previous_query}"""
        )
    ])

    chain = prompt | get_groq_llm() | StrOutputParser()

    new_query = await chain.ainvoke({
        "question": question,
        "previous_query": previous_query,
    })

    return {
        "standalone_query": new_query.strip(),
        "retrieval_attempt": state.get("retrieval_attempt", 0) + 1,
    }