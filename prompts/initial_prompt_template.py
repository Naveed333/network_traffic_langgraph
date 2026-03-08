"""
Prompt components for the initial (zero-shot / few-shot) traffic prediction.

Three building blocks are assembled by build_initial_prompt (NODE 1):
    p_exam  — few-shot demonstration of the prediction format
    p_input — formatted x_t input data
    p_ques  — task instruction and output format specification

The ChatPromptTemplate here is used in NODE 2 (initial_predict) via LCEL:
    INITIAL_PREDICTION_TEMPLATE | llm | StrOutputParser()
"""

from langchain_core.prompts import ChatPromptTemplate

# ── System persona ─────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert network traffic forecasting system. "
    "Your role is to analyse historical hourly traffic patterns and predict "
    "the traffic for every hour of the next day. "
    "Always output EXACTLY 24 comma-separated float values — one per hour "
    "(hour 0 through hour 23) — and nothing else. "
    "Do not include labels, units, explanations, or markdown formatting."
)

# ── ChatPromptTemplate (used in NODE 2 via LCEL) ──────────────────────────────

INITIAL_PREDICTION_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        (
            "human",
            "{p_exam}\n\n"
            "{p_input}\n\n"
            "{p_ques}",
        ),
    ]
)

# ── Few-shot example block (p_exam) ──────────────────────────────────────────

_FEW_SHOT_EXAMPLE = """\
## Demonstration — Example Traffic Prediction

### Input: previous-day hourly traffic (Mbps)
Hour 00: 462.10 | Hour 01: 391.45 | Hour 02: 325.80 | Hour 03: 293.20
Hour 04: 281.60 | Hour 05: 314.90 | Hour 06: 428.75 | Hour 07: 587.30
Hour 08: 731.20 | Hour 09: 788.50 | Hour 10: 810.40 | Hour 11: 826.70
Hour 12: 802.30 | Hour 13: 815.90 | Hour 14: 808.60 | Hour 15: 830.10
Hour 16: 857.40 | Hour 17: 882.30 | Hour 18: 864.20 | Hour 19: 831.50
Hour 20: 758.70 | Hour 21: 684.20 | Hour 22: 581.90 | Hour 23: 493.30

### Output: predicted traffic for the next day (24 values)
471.00, 398.50, 332.10, 299.40, 287.80, 321.60, 436.20, 596.80, 743.50, 801.30, 823.10, 839.40, 815.80, 829.20, 822.10, 844.60, 872.10, 897.80, 879.60, 847.30, 773.20, 697.40, 594.60, 503.80
"""


def build_p_exam() -> str:
    """Return the few-shot example block (p_exam)."""
    return _FEW_SHOT_EXAMPLE.strip()


def build_p_input(x_t: list, target_date: str) -> str:
    """
    Format the input traffic sequence as a structured text block (p_input).

    Args:
        x_t:         Previous-day hourly traffic (24 float values).
        target_date: ISO date string of the day being predicted.

    Returns:
        Formatted string ready for injection into the prompt.
    """
    lines = [
        f"## Input: Previous-Day Hourly Traffic",
        f"Target date to predict: {target_date}",
        "",
        "Hourly traffic values (Mbps):",
    ]
    for hour, val in enumerate(x_t):
        lines.append(f"  Hour {hour:02d}: {val:8.2f} Mbps")
    return "\n".join(lines)


def build_p_ques() -> str:
    """Return the task-instruction / question block (p_ques)."""
    return (
        "## Task\n"
        "Based on the previous-day traffic pattern shown above, predict the "
        "network traffic for each hour of the target date.\n\n"
        "Consider the following when forming your prediction:\n"
        "  1. Daily periodicity — traffic typically follows a 24-hour sine-like cycle.\n"
        "  2. Peak hours — expect peaks near the morning commute (~08:00) and "
        "evening commute (~17:00–18:00).\n"
        "  3. Overnight low — minimum traffic usually occurs between 02:00–05:00.\n"
        "  4. Day-over-day trend — traffic on the target day may be slightly higher "
        "or lower than the previous day.\n\n"
        "Respond with EXACTLY 24 comma-separated float values representing the "
        "predicted traffic (in Mbps) for hours 00 through 23, like this example:\n"
        "471.00, 398.50, 332.10, 299.40, ..., 503.80"
    )


# ── Dynamic p_exam from converged evaluation contexts ─────────────────────────

def build_p_exam_from_contexts(contexts: list) -> str:
    """
    Build a rich, dynamic p_exam from a list of converged evaluation contexts.

    Each context is the result dict saved by run_pipeline() for one historical
    day. Contexts are presented oldest → newest so the most recent lesson
    (most relevant) is closest to the deployment prompt.

    Token budget strategy:
        - Most recent context  → full detail (all iterations)
        - Older contexts       → condensed (initial + final only)

    Args:
        contexts: List of result dicts ordered oldest → newest.
                  Each dict must contain:
                    eval_date, x_t, ground_truth, y_hat_history,
                    pfeed_history, prefine_history, mae_history,
                    final_mae, converged, total_iterations

    Returns:
        Formatted string ready to replace the static p_exam in the prompt.
    """
    if not contexts:
        return _FEW_SHOT_EXAMPLE.strip()

    lines = [
        "## In-Context Learning — Historical Prediction Lessons",
        f"({len(contexts)} days evaluated; most recent lesson is most relevant)\n",
    ]

    for idx, ctx in enumerate(contexts):
        is_most_recent = (idx == len(contexts) - 1)
        day_num        = idx + 1
        eval_date      = ctx.get("eval_date", f"Day {day_num}")
        mae_history    = ctx.get("mae_history", [])
        y_hat_history  = ctx.get("y_hat_history", [])
        pfeed_history  = ctx.get("pfeed_history", [])
        prefine_history= ctx.get("prefine_history", [])
        x_t            = ctx.get("x_t", [])
        final_mae      = ctx.get("final_mae", None)
        total_iters    = ctx.get("total_iterations", len(y_hat_history) - 1)

        lines.append(f"{'─' * 60}")
        lines.append(
            f"### Lesson {day_num}: {eval_date} "
            f"({'Most Recent' if is_most_recent else 'Historical'})"
        )

        # Input data — always shown
        if x_t:
            x_t_str = " | ".join(
                f"H{h:02d}:{v:.1f}" for h, v in enumerate(x_t)
            )
            lines.append(f"**Input (previous day hourly Mbps):** {x_t_str}")

        if is_most_recent:
            # ── Full detail for most recent day ───────────────────────────
            for i, y_hat in enumerate(y_hat_history):
                label    = "Initial Prediction ŷ₀" if i == 0 else f"Refined Prediction ŷ{i}"
                mae_val  = f" → MAE: {mae_history[i]:.2f} Mbps" if i < len(mae_history) else ""
                vals_str = ", ".join(f"{v:.2f}" for v in y_hat)
                lines.append(f"**[Iter {i}] {label}:** {vals_str}{mae_val}")

                if i < len(pfeed_history):
                    # Truncate long feedback to first 300 chars for token budget
                    feed = pfeed_history[i][:300].replace("\n", " ")
                    lines.append(f"  ↳ Feedback: {feed}...")

                if i < len(prefine_history):
                    refine = prefine_history[i][:200].replace("\n", " ")
                    lines.append(f"  ↳ Refinement instruction: {refine}")
        else:
            # ── Condensed for older days (initial + final only) ───────────
            if y_hat_history:
                y0_str = ", ".join(f"{v:.2f}" for v in y_hat_history[0])
                lines.append(
                    f"**Initial ŷ₀:** {y0_str}"
                    + (f" → MAE: {mae_history[0]:.2f} Mbps" if mae_history else "")
                )
            if len(y_hat_history) > 1:
                yf_str = ", ".join(f"{v:.2f}" for v in y_hat_history[-1])
                lines.append(
                    f"**Final  ŷ{len(y_hat_history)-1}:** {yf_str}"
                    + (f" → MAE: {mae_history[-1]:.2f} Mbps" if mae_history else "")
                )
            if pfeed_history:
                feed = pfeed_history[-1][:200].replace("\n", " ")
                lines.append(f"  ↳ Key feedback: {feed}...")

        # Convergence summary — always shown
        converged_str = "✓ converged" if ctx.get("converged") else "✗ max iterations"
        lines.append(
            f"**Result:** {total_iters} refinement(s), "
            f"final MAE={final_mae:.2f} Mbps, {converged_str}"
        )
        lines.append("")

    lines.append(f"{'─' * 60}")
    lines.append(
        "Use the lessons above to guide your prediction. "
        "Pay close attention to Lesson "
        f"{len(contexts)} (most recent) as it reflects the latest traffic patterns.\n"
    )

    return "\n".join(lines)
