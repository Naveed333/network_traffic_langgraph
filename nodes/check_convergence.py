"""
NODE 7 — check_convergence  (Conditional node + router)

Pure Python node — NO LLM call.

Evaluates two stop conditions after every refinement:

    1. Iteration budget: iteration >= max_iterations
       → ensures the pipeline always terminates.

    2. MAE convergence: |MAE[-1] - MAE[-2]| < convergence_threshold
       → stops when further refinements yield diminishing returns.

``convergence_router`` is the routing function registered on the conditional
edge.  It returns ``"converged"`` (→ END) or ``"continue"`` (→ mae_feedback).
"""

import logging

from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger

logger: logging.Logger = setup_logger(__name__)


def check_convergence(state: TrafficPredictionState) -> dict:
    """
    NODE 7: Evaluate stop conditions and set ``converged`` flag.

    Args:
        state: Requires iteration, max_iterations, mae_history,
               convergence_threshold.

    Returns:
        Partial state update with ``converged`` flag.
    """
    log_node_entry(logger, "check_convergence", state["iteration"])

    iteration   = state["iteration"]
    max_iter    = state["max_iterations"]
    threshold   = state["convergence_threshold"]
    mae_history = state.get("mae_history", [])

    converged = False
    reason    = ""

    # ── Condition 1: iteration budget ─────────────────────────────────────────
    if iteration >= max_iter:
        converged = True
        reason    = (
            f"Max iterations reached: {iteration} >= {max_iter}"
        )

    # ── Condition 2: MAE convergence ──────────────────────────────────────────
    elif len(mae_history) >= 2:
        mae_delta = abs(mae_history[-1] - mae_history[-2])
        if mae_delta < threshold:
            converged = True
            reason    = (
                f"MAE converged: Δ={mae_delta:.6f} < threshold={threshold} "
                f"(MAE={mae_history[-1]:.4f})"
            )
        else:
            reason = (
                f"Not converged: Δ={mae_delta:.4f} >= threshold={threshold} "
                f"(iter {iteration}/{max_iter})"
            )
    else:
        reason = (
            f"Not enough MAE history ({len(mae_history)} point(s)) "
            f"to check convergence — continuing"
        )

    log_node_result(
        logger,
        "check_convergence",
        iteration,
        parsed_output=f"converged={converged} | {reason}",
    )

    return {"converged": converged}


def convergence_router(state: TrafficPredictionState) -> str:
    """
    LangGraph routing function for the conditional edge after check_convergence.

    Returns:
        ``"converged"`` if the pipeline should end, ``"continue"`` to loop back
        to mae_feedback for the next refinement iteration.
    """
    if state.get("converged", False):
        logger.info(
            "[ROUTER] converged=True → routing to END (iter=%d, MAE=%.4f)",
            state["iteration"],
            state["mae_history"][-1] if state.get("mae_history") else float("inf"),
        )
        return "converged"

    logger.info(
        "[ROUTER] converged=False → routing to mae_feedback (iter=%d)",
        state["iteration"],
    )
    return "continue"
