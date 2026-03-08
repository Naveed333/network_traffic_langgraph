"""
NODE 2 — api_call_initial_predict  (API CALL 1)

Sends the assembled initial prompt to GPT-4o via LCEL and parses the
24 comma-separated float values that constitute the first prediction ŷ₀.

Retry policy: up to MAX_RETRIES attempts with exponential back-off.
"""

import logging
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings
from prompts.initial_prompt_template import INITIAL_PREDICTION_TEMPLATE
from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger
from utils.parser import parse_24_floats
from utils.token_counter import count_tokens

logger: logging.Logger = setup_logger(__name__)

MAX_RETRIES: int = 3


def api_call_initial_predict(state: TrafficPredictionState) -> dict:
    """
    NODE 2: Send the initial prompt to the LLM and parse ŷ₀.

    LCEL chain: INITIAL_PREDICTION_TEMPLATE | llm | StrOutputParser()

    Args:
        state: Current pipeline state (requires p_exam, p_input, p_ques).

    Returns:
        Partial state update with y_hat_current, y_hat_history,
        api_calls_count, and total_tokens_used.
    """
    log_node_entry(logger, "initial_predict", state["iteration"])

    # ── Build LCEL chain ──────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.openai_api_key,
    )
    chain = INITIAL_PREDICTION_TEMPLATE | llm | StrOutputParser()

    # ── Token accounting ──────────────────────────────────────────────────────
    prompt_text = (
        f"{state['p_exam']}\n{state['p_input']}\n{state['p_ques']}"
    )
    token_count = count_tokens(prompt_text, settings.model_name)

    # ── Retry loop ────────────────────────────────────────────────────────────
    prediction: Optional[list] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chain.invoke(
                {
                    "p_exam":  state["p_exam"],
                    "p_input": state["p_input"],
                    "p_ques":  state["p_ques"],
                }
            )
            prediction = parse_24_floats(response)
            if prediction is not None:
                logger.debug(
                    "initial_predict: parsed successfully on attempt %d", attempt
                )
                break
            logger.warning(
                "initial_predict: parse failed on attempt %d/%d — retrying",
                attempt, MAX_RETRIES,
            )
        except Exception as exc:
            logger.error(
                "initial_predict: API error on attempt %d/%d — %s",
                attempt, MAX_RETRIES, exc,
            )
        if attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))   # 1 s, 2 s, 4 s …

    if prediction is None:
        logger.error(
            "initial_predict: all %d retries exhausted — using zero fallback",
            MAX_RETRIES,
        )
        prediction = [0.0] * 24

    # ── State update ──────────────────────────────────────────────────────────
    new_history = state.get("y_hat_history", []) + [prediction]

    log_node_result(
        logger,
        "initial_predict",
        state["iteration"],
        token_count=token_count,
        parsed_output=f"[{prediction[0]:.2f}, …, {prediction[-1]:.2f}]",
    )

    return {
        "y_hat_current":   prediction,
        "y_hat_history":   new_history,
        "api_calls_count": state.get("api_calls_count", 0) + 1,
        "total_tokens_used": state.get("total_tokens_used", 0) + token_count,
    }
