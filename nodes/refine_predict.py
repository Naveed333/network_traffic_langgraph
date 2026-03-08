"""
NODE 6 — api_call_refine  (API CALL 4 / 7 / 10 / …)

The core iterative refinement call.

Before every LLM invocation this node:
    1. Assembles the full context (x_t + prediction history + feedback history
       + refinement instruction).
    2. Counts tokens via tiktoken.
    3. If tokens > 80 % of the context window, summarises history to the most
       recent two iterations and recounts.
    4. Sends the (possibly summarised) prompt to GPT-4o.
    5. Parses the 24-value prediction, increments the iteration counter, and
       appends ŷᵢ₊₁ to y_hat_history.

Retry policy: up to MAX_RETRIES attempts with exponential back-off.

LCEL chain: REFINEMENT_TEMPLATE | llm | StrOutputParser()
"""

import logging
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings
from prompts.refinement_template import (
    REFINEMENT_TEMPLATE,
    build_feedback_history,
    build_prediction_history,
    build_prefine_instruction,
)
from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger
from utils.parser import parse_24_floats
from utils.token_counter import check_context_guard, count_tokens

logger: logging.Logger = setup_logger(__name__)

MAX_RETRIES: int = 3


# ── History summarisation (context-guard helper) ──────────────────────────────

def _summarise_history(
    y_hat_history: list,
    pfeed_history: list,
    keep_last: int = 2,
) -> tuple[str, str]:
    """
    Reduce history to the most recent *keep_last* iterations to stay within
    the context-window budget.

    Args:
        y_hat_history: All prediction lists.
        pfeed_history: All feedback strings.
        keep_last:     Number of most-recent entries to retain.

    Returns:
        (prediction_history_str, feedback_history_str)
    """
    total = len(y_hat_history)

    truncated_preds  = y_hat_history[-keep_last:]
    truncated_feeds  = pfeed_history[-keep_last:]

    pred_str  = build_prediction_history(truncated_preds)
    feed_str  = build_feedback_history(truncated_feeds)

    if total > keep_last:
        pred_str = (
            f"[History summarised — showing last {keep_last} of {total} iterations]\n"
            + pred_str
        )
        feed_str = (
            f"[History summarised — showing last {keep_last} of {total} feedbacks]\n"
            + feed_str
        )

    return pred_str, feed_str


# ── Node ──────────────────────────────────────────────────────────────────────

def api_call_refine(state: TrafficPredictionState) -> dict:
    """
    NODE 6: Iterative refinement LLM call with context-window guard.

    LCEL chain: REFINEMENT_TEMPLATE | llm | StrOutputParser()

    Args:
        state: Full pipeline state.

    Returns:
        Partial state update with y_hat_current, y_hat_history, iteration,
        prefine_current, api_calls_count, total_tokens_used.
    """
    log_node_entry(logger, "refine_predict", state["iteration"])

    x_t_str              = ", ".join(f"{v:.2f}" for v in state["x_t"])
    y_hat_history        = state.get("y_hat_history", [])
    pfeed_history        = state.get("pfeed_history", [])
    mae_history          = state.get("mae_history", [])
    prefine_instruction  = build_prefine_instruction(mae_history, state["iteration"])

    # ── Initial history strings ───────────────────────────────────────────────
    prediction_history = build_prediction_history(y_hat_history)
    feedback_history   = build_feedback_history(pfeed_history)

    # ── Context window guard ──────────────────────────────────────────────────
    full_context = (
        f"{x_t_str}\n{prediction_history}\n{feedback_history}\n{prefine_instruction}"
    )
    within_budget, token_count = check_context_guard(
        full_context,
        settings.context_window_limit,
        settings.context_window_guard_pct,
        settings.model_name,
    )

    if not within_budget:
        logger.warning(
            "refine_predict: context guard triggered (%d tokens) — summarising history",
            token_count,
        )
        prediction_history, feedback_history = _summarise_history(
            y_hat_history, pfeed_history, keep_last=2
        )
        full_context = (
            f"{x_t_str}\n{prediction_history}\n{feedback_history}\n{prefine_instruction}"
        )
        _, token_count = check_context_guard(
            full_context,
            settings.context_window_limit,
            guard_pct=1.0,   # just count; already summarised
            model=settings.model_name,
        )
        logger.info(
            "refine_predict: after summarisation — %d tokens", token_count
        )

    # ── Build LCEL chain ──────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.openai_api_key,
    )
    chain = REFINEMENT_TEMPLATE | llm | StrOutputParser()

    # ── Retry loop ────────────────────────────────────────────────────────────
    prediction: Optional[list] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chain.invoke(
                {
                    "x_t":                x_t_str,
                    "prediction_history": prediction_history,
                    "feedback_history":   feedback_history,
                    "prefine_instruction": prefine_instruction,
                }
            )
            prediction = parse_24_floats(response)
            if prediction is not None:
                logger.debug(
                    "refine_predict: parsed successfully on attempt %d", attempt
                )
                break
            logger.warning(
                "refine_predict: parse failed on attempt %d/%d — retrying",
                attempt, MAX_RETRIES,
            )
        except Exception as exc:
            logger.error(
                "refine_predict: API error on attempt %d/%d — %s",
                attempt, MAX_RETRIES, exc,
            )
        if attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))

    # ── Fallback: keep current prediction ─────────────────────────────────────
    if prediction is None:
        logger.error(
            "refine_predict: all %d retries exhausted — keeping current prediction",
            MAX_RETRIES,
        )
        prediction = list(state["y_hat_current"])

    # ── Increment iteration, update history ───────────────────────────────────
    new_iteration = state["iteration"] + 1
    new_history   = list(y_hat_history) + [prediction]

    log_node_result(
        logger,
        "refine_predict",
        new_iteration,
        token_count=token_count,
        parsed_output=f"[{prediction[0]:.2f}, …, {prediction[-1]:.2f}]",
        new_iter=new_iteration,
    )

    return {
        "y_hat_current":     prediction,
        "y_hat_history":     new_history,
        "iteration":         new_iteration,
        "prefine_current":   prefine_instruction,
        "api_calls_count":   state.get("api_calls_count", 0) + 1,
        "total_tokens_used": state.get("total_tokens_used", 0) + token_count,
    }
