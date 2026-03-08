"""
NODE 5 — assemble_feedback

Pure Python node — NO LLM call.

Combines the MAE score and sine-fit formula strings produced by NODEs 3 & 4
into a single structured feedback block (pfeed_current) that is:
    - Stored in pfeed_history for use by subsequent refinement iterations.
    - Injected into the refinement prompt so the LLM can see a compact summary
      of its errors alongside the symbolic pattern discrepancy.
"""

import logging

from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger

logger: logging.Logger = setup_logger(__name__)


def assemble_feedback(state: TrafficPredictionState) -> dict:
    """
    NODE 5: Build the pfeed string from MAE + sine-curve results.

    Args:
        state: Requires mae_score, mae_history, sine_fit_actual,
               sine_fit_predicted, iteration.

    Returns:
        Partial state update with pfeed_current and pfeed_history.
    """
    log_node_entry(logger, "assemble_feedback", state["iteration"])

    iteration    = state["iteration"]
    mae          = state.get("mae_score", float("inf"))
    mae_history  = state.get("mae_history", [])
    f_act        = state.get("sine_fit_actual",    "N/A")
    f_pred       = state.get("sine_fit_predicted", "N/A")

    # ── Compute improvement delta (for display) ───────────────────────────────
    if len(mae_history) >= 2:
        delta       = mae_history[-1] - mae_history[-2]
        delta_str   = f"{delta:+.4f} Mbps vs previous"
        trend_emoji = "↓ improving" if delta < 0 else ("↑ worsening" if delta > 0 else "→ unchanged")
    else:
        delta_str   = "first evaluation"
        trend_emoji = "→ baseline"

    # ── Derive amplitude / phase / offset observations ────────────────────────
    f_act_clean  = f_act  if f_act  else "unavailable"
    f_pred_clean = f_pred if f_pred else "unavailable"

    # ── Assemble the structured feedback block ────────────────────────────────
    pfeed = f"""\
=== Feedback — Iteration {iteration} ===

[Error Metrics]
  Current MAE  : {mae:.4f} Mbps  ({trend_emoji}, {delta_str})
  MAE history  : {[f"{m:.4f}" for m in mae_history]}

[Periodic Pattern Analysis — Sine/Cosine Fit]
  f_act  (actual)   : {f_act_clean}
  f_pred (predicted): {f_pred_clean}

[Interpretation]
  • If amplitude A_actual ≠ A_pred  → your prediction over- or under-estimates traffic peaks.
  • If phase    φ_actual ≠ φ_pred   → your predicted peaks/troughs are shifted in time.
  • If offset   C_actual ≠ C_pred   → your baseline traffic level is too high or too low.
  • Focus on the hours with the largest absolute residuals in the next iteration.
"""

    new_pfeed_history = state.get("pfeed_history", []) + [pfeed]

    log_node_result(
        logger,
        "assemble_feedback",
        state["iteration"],
        parsed_output=f"pfeed assembled ({len(pfeed)} chars)",
        mae=f"{mae:.4f}",
    )

    return {
        "pfeed_current": pfeed,
        "pfeed_history": new_pfeed_history,
    }
