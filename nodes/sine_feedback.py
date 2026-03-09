"""
NODE 4 — api_call_sine_feedback  (API CALL 3 / 6 / 9 / …)

Asks the LLM to fit sine/cosine curves to both the ground-truth traffic and
the current prediction, then returns the formula strings as f_act and f_pred.

These are used in NODE 5 to build a structured feedback block that highlights
amplitude, frequency, and phase-shift discrepancies.

Fallback: If parsing fails, the local SciPy-based ``compute_sine_fits``
function from evaluation/sine_fit.py is used instead.

LCEL chain: SINE_FIT_TEMPLATE | llm | StrOutputParser()
"""

import logging
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings
from evaluation.sine_fit import compute_sine_fits
from prompts.feedback_template import SINE_FIT_TEMPLATE
from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger
from utils.parser import parse_sine_strings
from utils.token_counter import count_tokens

logger: logging.Logger = setup_logger(__name__)

MAX_RETRIES: int = 3


def api_call_sine_feedback(state: TrafficPredictionState) -> dict:
    """
    NODE 4: LLM-based sine/cosine curve fitting for pattern feedback.

    LCEL chain: SINE_FIT_TEMPLATE | llm | StrOutputParser()

    Args:
        state: Requires ground_truth, y_hat_current.

    Returns:
        Partial state update with sine_fit_actual, sine_fit_predicted,
        api_calls_count, total_tokens_used.
    """
    log_node_entry(logger, "sine_feedback", state["iteration"])

    actual_str    = ", ".join(f"{v:.2f}" for v in state["ground_truth"])
    predicted_str = ", ".join(f"{v:.2f}" for v in state["y_hat_current"])
    token_count   = count_tokens(
        f"{actual_str}\n{predicted_str}", settings.model_name
    )

    # ── Build LCEL chain ──────────────────────────────────────────────────────
    # max_tokens raised to 2048 — reasoning models spend tokens internally;
    # 512 was too low and caused truncated / empty responses.
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=0.1,
        max_tokens=2048,
        api_key=settings.openai_api_key,
    )
    chain = SINE_FIT_TEMPLATE | llm | StrOutputParser()

    # ── Retry loop ────────────────────────────────────────────────────────────
    f_act:  Optional[str] = None
    f_pred: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chain.invoke(
                {
                    "actual_values":    actual_str,
                    "predicted_values": predicted_str,
                }
            )
            # Log raw response at DEBUG so format mismatches are visible
            logger.debug(
                "sine_feedback: raw response (attempt %d): %r", attempt, response[:400]
            )
            f_act, f_pred = parse_sine_strings(response)
            if f_act and f_pred:
                logger.debug(
                    "sine_feedback: parsed successfully on attempt %d", attempt
                )
                break
            logger.warning(
                "sine_feedback: parse failed on attempt %d/%d — retrying "
                "(raw snippet: %r)",
                attempt, MAX_RETRIES, response[:150],
            )
        except Exception as exc:
            logger.error(
                "sine_feedback: API error on attempt %d/%d — %s",
                attempt, MAX_RETRIES, exc,
            )
        if attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))

    # ── Local fallback ────────────────────────────────────────────────────────
    if not f_act or not f_pred:
        logger.warning("sine_feedback: falling back to local compute_sine_fits")
        f_act, f_pred = compute_sine_fits(
            state["ground_truth"], state["y_hat_current"]
        )

    log_node_result(
        logger,
        "sine_feedback",
        state["iteration"],
        token_count=token_count,
        parsed_output=f"f_act={f_act[:60]}…",
    )

    return {
        "sine_fit_actual":    f_act,
        "sine_fit_predicted": f_pred,
        "api_calls_count":    state.get("api_calls_count", 0) + 1,
        "total_tokens_used":  state.get("total_tokens_used", 0) + token_count,
    }
