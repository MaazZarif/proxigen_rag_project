
from contextlib import asynccontextmanager
import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from proxigen_rag.ingestion.vector_store import initialize_collection
from proxigen_rag.services.checkpointer import get_checkpointer
from proxigen_rag.graph.graph import build_graph
from proxigen_rag.api.chat import router as chat_router
from dotenv import load_dotenv
from proxigen_rag.ingestion.vector_store import get_vector_store

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await initialize_collection()
        get_vector_store()

        async with get_checkpointer() as checkpointer:
            await checkpointer.setup()
            app.state.graph = build_graph(checkpointer)

            logger.info("Proxigen RAG API started successfully")

            yield

    except Exception as e:
        logger.exception("Application startup or runtime failed: %s", e)
        raise

    finally:
        logger.info("Proxigen RAG API shutting down")


app = FastAPI(
    title="Proxigen RAG API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
