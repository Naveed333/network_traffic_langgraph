"""
NODE 3 — api_call_mae_feedback  (API CALL 2 / 5 / 8 / …)

Asks the LLM to compute the Mean Absolute Error (MAE) between the current
prediction and ground truth. The LLM is given the explicit formula and both
value sequences so it can surface awareness of the numeric error to the model.

Fallback: If the LLM response cannot be parsed, the local ``compute_mae``
function from evaluation/mae.py is used instead.

LCEL chain: MAE_CALCULATION_TEMPLATE | llm | StrOutputParser()
"""

import logging
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings
from evaluation.mae import compute_mae
from prompts.feedback_template import MAE_CALCULATION_TEMPLATE
from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger
from utils.parser import parse_float
from utils.token_counter import count_tokens

logger: logging.Logger = setup_logger(__name__)

MAX_RETRIES: int = 3


def api_call_mae_feedback(state: TrafficPredictionState) -> dict:
    """
    NODE 3: LLM-based MAE feedback computation.

    LCEL chain: MAE_CALCULATION_TEMPLATE | llm | StrOutputParser()

    Args:
        state: Requires ground_truth, y_hat_current.

    Returns:
        Partial state update with mae_score, mae_history,
        api_calls_count, total_tokens_used.
    """
    log_node_entry(logger, "mae_feedback", state["iteration"])

    actual_str    = ", ".join(f"{v:.2f}" for v in state["ground_truth"])
    predicted_str = ", ".join(f"{v:.2f}" for v in state["y_hat_current"])
    token_count   = count_tokens(
        f"{actual_str}\n{predicted_str}", settings.model_name
    )

    # ── Build LCEL chain (temperature=0 for deterministic math) ──────────────
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        max_tokens=256,
        api_key=settings.openai_api_key,
    )
    chain = MAE_CALCULATION_TEMPLATE | llm | StrOutputParser()

    # ── Retry loop ────────────────────────────────────────────────────────────
    mae_score: Optional[float] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chain.invoke(
                {
                    "actual_values":    actual_str,
                    "predicted_values": predicted_str,
                    "num_points":       24,
                }
            )
            mae_score = parse_float(response)
            if mae_score is not None:
                logger.debug(
                    "mae_feedback: parsed MAE=%.4f on attempt %d", mae_score, attempt
                )
                break
            logger.warning(
                "mae_feedback: parse failed on attempt %d/%d — retrying",
                attempt, MAX_RETRIES,
            )
        except Exception as exc:
            logger.error(
                "mae_feedback: API error on attempt %d/%d — %s",
                attempt, MAX_RETRIES, exc,
            )
        if attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))

    # ── Local fallback ────────────────────────────────────────────────────────
    if mae_score is None:
        logger.warning("mae_feedback: falling back to local compute_mae")
        mae_score = compute_mae(state["ground_truth"], state["y_hat_current"])

    new_mae_history = state.get("mae_history", []) + [mae_score]

    log_node_result(
        logger,
        "mae_feedback",
        state["iteration"],
        token_count=token_count,
        parsed_output=f"MAE={mae_score:.4f}",
        mae_history=str([f"{m:.4f}" for m in new_mae_history]),
    )

    return {
        "mae_score":         mae_score,
        "mae_history":       new_mae_history,
        "api_calls_count":   state.get("api_calls_count", 0) + 1,
        "total_tokens_used": state.get("total_tokens_used", 0) + token_count,
    }
