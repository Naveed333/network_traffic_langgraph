"""
NODE 1 — build_initial_prompt

Pure Python node (no LLM call).
Assembles the three prompt components (p_exam, p_input, p_ques) from the
state's x_t input data and target_date, then stores them back in state.

These components are consumed by NODE 2 (initial_predict) via the
INITIAL_PREDICTION_TEMPLATE ChatPromptTemplate.
"""

import logging

from prompts.initial_prompt_template import build_p_exam, build_p_input, build_p_ques
from state import TrafficPredictionState
from utils.logger import log_node_entry, log_node_result, setup_logger

logger: logging.Logger = setup_logger(__name__)


def build_initial_prompt(state: TrafficPredictionState) -> dict:
    """
    Build the three prompt blocks and inject them into state.

    Args:
        state: Current pipeline state (requires x_t, target_date).

    Returns:
        Partial state update dict with p_exam, p_input, p_ques populated.
    """
    log_node_entry(logger, "build_initial_prompt", state["iteration"])

    p_exam  = build_p_exam()
    p_input = build_p_input(state["x_t"], state["target_date"])
    p_ques  = build_p_ques()

    log_node_result(
        logger,
        "build_initial_prompt",
        state["iteration"],
        parsed_output=(
            f"p_exam={len(p_exam)} chars | "
            f"p_input={len(p_input)} chars | "
            f"p_ques={len(p_ques)} chars"
        ),
    )

    return {
        "p_exam":  p_exam,
        "p_input": p_input,
        "p_ques":  p_ques,
    }
