"""
NODE 4 — api_call_sine_feedback

Fits sine/cosine curves to both the ground-truth traffic and the current
prediction, returning formula strings f_act and f_pred used by NODE 5 to
build structured pattern feedback.

Curve fitting is always done locally via SciPy (evaluation/sine_fit.py).
Reasoning models (gpt-5-mini, o-series) consume their internal token budget
on chain-of-thought and returned empty visible output on the first 1–2
attempts each time; SciPy curve fitting is deterministic, instant, and
produces the same parametric output format the LLM expects in its prompt.

The formula strings are embedded in the pfeed block the LLM reads during
every refinement call — so amplitude/phase discrepancies are always surfaced.
"""

import logging

from evaluation.sine_fit import compute_sine_fits
from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger
from utils.token_counter import count_tokens
from config import settings

logger: logging.Logger = setup_logger(__name__)


def api_call_sine_feedback(state: TrafficPredictionState) -> dict:
    """
    NODE 4: Fit sine curves locally and record formula strings in state.

    Args:
        state: Requires ground_truth, y_hat_current.

    Returns:
        Partial state update with sine_fit_actual, sine_fit_predicted,
        total_tokens_used.
        api_calls_count is NOT incremented (no LLM call is made).
    """
    log_node_entry(logger, "sine_feedback", state["iteration"])

    actual_str    = ", ".join(f"{v:.2f}" for v in state["ground_truth"])
    predicted_str = ", ".join(f"{v:.2f}" for v in state["y_hat_current"])
    token_count   = count_tokens(
        f"{actual_str}\n{predicted_str}", settings.model_name
    )

    # ── Local computation (always) ────────────────────────────────────────────
    f_act, f_pred = compute_sine_fits(
        state["ground_truth"], state["y_hat_current"]
    )

    log_node_result(
        logger,
        "sine_feedback",
        state["iteration"],
        token_count=token_count,
        parsed_output=f"f_act={f_act[:50]}… | f_pred={f_pred[:50]}…",
    )

    return {
        "sine_fit_actual":    f_act,
        "sine_fit_predicted": f_pred,
        "total_tokens_used":  state.get("total_tokens_used", 0) + token_count,
    }
