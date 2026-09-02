import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Single source of truth for which LLM the whole app uses.
# To swap models or providers later, change this file — not every
# file that happens to need an LLM.

# fallback model name used when LLM_MODEL isn't set in .env
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# fallback Gemini thinking-budget setting used when LLM_THINKING_LEVEL isn't set in .env
DEFAULT_THINKING_LEVEL = "low"


# factory function: builds and returns the single shared LLM instance used everywhere in the app
def get_llm():
    """
    Returns the chat model used across the app (main chat, guardrail
    check, and the RAGAS evaluator via evaluation.py).

    Reads LLM_MODEL / LLM_THINKING_LEVEL from .env if set, otherwise
    falls back to the defaults above. This means you can override the
    model for a single run without editing code, e.g.:
        LLM_MODEL=gemini-2.5-flash python langgraph_backend.py
    """
    model_name = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    thinking_level = os.getenv("LLM_THINKING_LEVEL", DEFAULT_THINKING_LEVEL)

    return ChatGoogleGenerativeAI(model=model_name, thinking_level=thinking_level)