import os
from typing import Any

from langchain_groq import ChatGroq


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please configure your Groq API key before using the chat agents.")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0,
    )


def get_llm_with_tools(tools: list[Any]):
    llm = _get_llm()
    return llm.bind_tools(tools)
