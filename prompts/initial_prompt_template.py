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
