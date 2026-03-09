"""
LangChain PromptTemplates for the two feedback API calls:

    1. MAE_CALCULATION_TEMPLATE  — asks the LLM to compute and return MAE
       (NODE 3 / api_call_mae_feedback).

    2. SINE_FIT_TEMPLATE          — asks the LLM to fit sine/cosine curves to
       both actual and predicted traffic sequences and return the formulas
       (NODE 4 / api_call_sine_feedback).

Both templates are used via LCEL: template | llm | StrOutputParser()
"""

from langchain_core.prompts import PromptTemplate

# ── MAE calculation template ──────────────────────────────────────────────────

MAE_CALCULATION_TEMPLATE = PromptTemplate(
    input_variables=["actual_values", "predicted_values", "num_points"],
    template=(
        "You are a precise numerical computation assistant.\n\n"
        "Compute the Mean Absolute Error (MAE) between the following two sequences "
        "of network traffic values.\n\n"
        "Formula: MAE = (1 / n) × Σ |actual_i − predicted_i|  where n = {num_points}\n\n"
        "Actual traffic (Mbps, {num_points} hourly values):\n"
        "{actual_values}\n\n"
        "Predicted traffic (Mbps, {num_points} hourly values):\n"
        "{predicted_values}\n\n"
        "Instructions:\n"
        "  1. For each hour i, compute |actual_i − predicted_i|.\n"
        "  2. Sum all absolute differences.\n"
        "  3. Divide by {num_points}.\n\n"
        "Respond with ONLY the single numeric MAE value rounded to 4 decimal places.\n"
        "Example response: 47.3821\n"
        "Do NOT include units, text, or any other content."
    ),
)

# ── Sine / cosine fitting template ────────────────────────────────────────────
# Output is requested as JSON so parse_sine_strings can reliably extract
# f_act and f_pred regardless of model verbosity or formatting style.

SINE_FIT_TEMPLATE = PromptTemplate(
    input_variables=["actual_values", "predicted_values"],
    template=(
        "You are an expert signal-processing assistant specialising in fitting "
        "periodic functions to time-series data.\n\n"
        "Fit a sine model of the form:\n"
        "    f(t) = A * sin(omega * t + phi) + C\n"
        "to each of the two 24-point (t = 0 to 23) network traffic sequences "
        "given below.\n\n"
        "Parameter definitions:\n"
        "  A     = amplitude       (half the peak-to-trough range, always >= 0)\n"
        "  omega = angular freq    (for a 24-hour period: omega = 2*pi/24 = 0.2618)\n"
        "  phi   = phase shift     (radians; adjust so the peak aligns with data)\n"
        "  C     = vertical offset (mean of the sequence)\n\n"
        "Actual traffic (Mbps, hours 0-23):\n"
        "{actual_values}\n\n"
        "Predicted traffic (Mbps, hours 0-23):\n"
        "{predicted_values}\n\n"
        "Instructions:\n"
        "  1. Compute A as half the (max - min) of the sequence.\n"
        "  2. Set C as the mean of the sequence.\n"
        "  3. Set omega = 0.2618 (fixed daily cycle).\n"
        "  4. Estimate phi so the sine peak aligns with the data peak.\n"
        "  5. Format each formula as: A * sin(omega * t + phi) + C\n"
        "     using only plain ASCII characters and * for multiplication.\n\n"
        "Respond with ONLY valid JSON — no extra text, no markdown fences:\n"
        '{{"f_act": "<formula for actual>", "f_pred": "<formula for predicted>"}}\n\n'
        "Example response:\n"
        '{{"f_act": "210.34 * sin(0.2618 * t - 1.5708) + 623.45", '
        '"f_pred": "195.72 * sin(0.2618 * t - 1.4320) + 605.88"}}'
    ),
)
