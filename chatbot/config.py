
"""
Central configuration for the FinSight chatbot.

Loads environment variables and provides the configured
Groq chat model for LangChain and LangGraph.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables from the project root .env file.
load_dotenv(override=True)


def get_chat_model():
    """
    Initialise and return the configured Groq chat model.
    """

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    # Log only whether the key is configured; never print any part of a secret.
    print("Groq API key configured:", bool(api_key))

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    if not model_name:
        raise RuntimeError(
            "GROQ_MODEL is missing from the configuration."
        )

    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )
