
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
import os

load_dotenv()

CHECKPOINT_DB_URL = os.getenv("CHECKPOINT_DB_URL")

if not CHECKPOINT_DB_URL:
    raise ValueError("CHECKPOINT_DB_URL is not set")


def get_checkpointer():
    return AsyncPostgresSaver.from_conn_string(
        CHECKPOINT_DB_URL
    )
