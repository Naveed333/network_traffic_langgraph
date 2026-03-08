"""
ChatPromptTemplate and helper functions for iterative prediction refinement
(NODE 6 / api_call_refine).

The refinement prompt is context-rich:
    - Original x_t input
    - Full y_hat_history (all previous predictions)
    - Full pfeed_history  (MAE + sine-fit feedback for each iteration)
    - A targeted prefine_instruction derived from the MAE trend

Used via LCEL: REFINEMENT_TEMPLATE | llm | StrOutputParser()
"""

from langchain_core.prompts import ChatPromptTemplate

# ── System persona ─────────────────────────────────────────────────────────────

_REFINEMENT_SYSTEM = (
    "You are an iterative network traffic forecasting engine. "
    "You receive your previous predictions alongside structured feedback that "
    "quantifies where and why those predictions were inaccurate. "
    "Your task is to produce an improved 24-hour traffic prediction by carefully "
    "incorporating the feedback. "
    "Always output EXACTLY 24 comma-separated float values — one per hour "
    "(hour 0 through hour 23) — and nothing else."
)

# ── Refinement ChatPromptTemplate ─────────────────────────────────────────────

REFINEMENT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _REFINEMENT_SYSTEM),
        (
            "human",
            "## Original Input Data (previous-day hourly traffic)\n"
            "{x_t}\n\n"
            "## Prediction History (all previous iterations)\n"
            "{prediction_history}\n\n"
            "## Feedback History (MAE + sine-curve analysis per iteration)\n"
            "{feedback_history}\n\n"
            "## Refinement Instruction\n"
            "{prefine_instruction}\n\n"
            "## Your Task\n"
            "Produce an improved prediction for the target day that addresses "
            "the errors identified in the feedback. "
            "Pay special attention to:\n"
            "  • Peak-hour timing and magnitude (morning ~08:00, evening ~17:00–18:00)\n"
            "  • Amplitude discrepancies highlighted by the sine-curve comparison\n"
            "  • Phase shifts that cause your peaks/troughs to be early or late\n"
            "  • Overnight baseline accuracy (hours 00–05)\n\n"
            "Respond with EXACTLY 24 comma-separated float values for hours 00–23:",
        ),
    ]
)


# ── History formatters ────────────────────────────────────────────────────────

def build_prediction_history(y_hat_history: list) -> str:
    """
    Render all previous predictions as a numbered list.

    Args:
        y_hat_history: List of 24-value float lists.

    Returns:
        Multi-line formatted string, or a placeholder if empty.
    """
    if not y_hat_history:
        return "(no previous predictions)"

    lines = []
    for i, pred in enumerate(y_hat_history):
        label = "Initial prediction" if i == 0 else f"Refinement {i}"
        values = ", ".join(f"{v:.2f}" for v in pred)
        lines.append(f"  [{i}] {label}: {values}")
    return "\n".join(lines)


def build_feedback_history(pfeed_history: list) -> str:
    """
    Render all previous feedback blocks.

    Args:
        pfeed_history: List of structured feedback strings.

    Returns:
        Concatenated feedback string with separators.
    """
    if not pfeed_history:
        return "(no previous feedback)"

    parts = []
    for i, feedback in enumerate(pfeed_history):
        parts.append(f"─── Feedback for iteration {i} ───\n{feedback}")
    return "\n\n".join(parts)


def build_prefine_instruction(mae_history: list, iteration: int) -> str:
    """
    Generate a targeted refinement instruction based on MAE trend.

    Args:
        mae_history: List of MAE values accumulated across iterations.
        iteration:   Current iteration index (before incrementing).

    Returns:
        A natural-language instruction string.
    """
    if not mae_history:
        return (
            "This is your first refinement. Focus on matching the overall "
            "daily traffic shape — especially peak timing and overnight baseline."
        )

    current_mae = mae_history[-1]

    if len(mae_history) < 2:
        return (
            f"Current MAE: {current_mae:.4f} Mbps. "
            "Refine your prediction by better aligning the predicted sine-curve "
            "parameters (amplitude A, phase φ, offset C) with the actual traffic pattern."
        )

    prev_mae  = mae_history[-2]
    mae_delta = current_mae - prev_mae

    if mae_delta < 0:
        return (
            f"Your last refinement improved MAE by {abs(mae_delta):.4f} Mbps "
            f"(now {current_mae:.4f} Mbps). Continue this direction: "
            "refine the amplitude and phase adjustments you made. "
            "Focus on the hours with the largest remaining residuals, "
            "particularly around peak and transition hours."
        )
    elif mae_delta == 0:
        return (
            f"MAE is unchanged at {current_mae:.4f} Mbps. "
            "Try a different strategy — re-examine the phase shift (φ) difference "
            "between f_act and f_pred, and adjust your peak-hour timing accordingly."
        )
    else:
        return (
            f"Your last refinement worsened MAE by {mae_delta:.4f} Mbps "
            f"(now {current_mae:.4f} Mbps). Roll back and try again: "
            "compare f_act vs f_pred carefully — check whether your amplitude "
            "is over- or under-estimated, and whether your phase is shifted too far."
        )
