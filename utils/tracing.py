"""
LangSmith tracing setup.

Call setup_tracing() once at the start of run_pipeline().
LangChain then automatically traces every LLM call, prompt, chain,
and LangGraph node — no changes needed anywhere else.
"""

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing() -> bool:
    """
    Activate LangSmith tracing by pushing the three required env-vars into
    os.environ (LangChain reads them from there at call-time).

    Returns:
        True if tracing was enabled, False if LANGCHAIN_API_KEY is missing.
    """
    from config import settings                         # local import avoids circular

    if not settings.langchain_api_key:
        logger.info("LangSmith tracing DISABLED (LANGCHAIN_API_KEY not set)")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
    os.environ["LANGCHAIN_API_KEY"]    = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"]    = settings.langchain_project

    logger.info(
        "LangSmith tracing ENABLED | project=%r | endpoint=https://smith.langchain.com",
        settings.langchain_project,
    )
    return True
