"""
NODE 3 — api_call_mae_feedback

Computes the Mean Absolute Error (MAE) between the current prediction and
ground truth and records it in state.

MAE is always computed locally via evaluation/mae.py — deterministic,
instant, and model-agnostic.  Reasoning models (gpt-5-mini, o-series) spent
all of their internal token budget on chain-of-thought and returned empty
visible output 100 % of the time when asked to compute MAE numerically;
local computation is identical in accuracy and faster.

The MAE value is passed to NODE 5 (assemble_feedback) which embeds it in the
structured feedback prompt the LLM reads during refinement — so the model
still "sees" its error score every iteration.
"""

import logging

from evaluation.mae import compute_mae
from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger
from utils.token_counter import count_tokens
from config import settings

logger: logging.Logger = setup_logger(__name__)


def api_call_mae_feedback(state: TrafficPredictionState) -> dict:
    """
    NODE 3: Compute MAE locally and record it in state.

    Args:
        state: Requires ground_truth, y_hat_current.

    Returns:
        Partial state update with mae_score, mae_history, total_tokens_used.
        api_calls_count is NOT incremented (no LLM call is made).
    """
    log_node_entry(logger, "mae_feedback", state["iteration"])

    actual_str    = ", ".join(f"{v:.2f}" for v in state["ground_truth"])
    predicted_str = ", ".join(f"{v:.2f}" for v in state["y_hat_current"])
    token_count   = count_tokens(
        f"{actual_str}\n{predicted_str}", settings.model_name
    )

    # ── Local computation (always) ────────────────────────────────────────────
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
        "total_tokens_used": state.get("total_tokens_used", 0) + token_count,
    }
