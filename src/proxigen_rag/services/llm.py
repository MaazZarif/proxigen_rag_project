

from functools import lru_cache
import os

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
   return ChatOllama(
        model="qwen3:4b",
        temperature=0,
    )

@lru_cache(maxsize=1)
def get_groq_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=3,
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_heavy_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_retries=3,
        timeout=60,
    )




# @lru_cache(maxsize=1)
# def get_groq_llm():
#     return ChatOpenAI(
#         model="gpt-oss-120b",
#         api_key=os.getenv("CEREBRAS_API_KEY"),
#         base_url="https://api.cerebras.ai/v1",
#         temperature=0,
#     )

# @lru_cache(maxsize=1)
# def get_heavy_llm():
#     return ChatOpenAI(
#         model="gpt-oss-120b",
#         api_key=os.getenv("CEREBRAS_API_KEY"),
#         base_url="https://api.cerebras.ai/v1",
#         temperature=0,
#     )