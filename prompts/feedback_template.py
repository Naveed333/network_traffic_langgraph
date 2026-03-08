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

SINE_FIT_TEMPLATE = PromptTemplate(
    input_variables=["actual_values", "predicted_values"],
    template=(
        "You are an expert signal-processing assistant specialising in fitting "
        "periodic functions to time-series data.\n\n"
        "Fit a sine/cosine model of the form:\n"
        "    f(t) = A × sin(ω × t + φ) + C\n"
        "to each of the two 24-point (t = 0 … 23) network traffic sequences "
        "given below.\n\n"
        "Parameter definitions:\n"
        "  A  = amplitude      (half the peak-to-trough range)\n"
        "  ω  = angular freq   (for a 24-hour period: ω ≈ 2π/24 ≈ 0.2618 rad/h)\n"
        "  φ  = phase shift    (radians; adjust so the model peak aligns with data)\n"
        "  C  = vertical offset (mean of the sequence)\n\n"
        "Actual traffic (Mbps, hours 0–23):\n"
        "{actual_values}\n\n"
        "Predicted traffic (Mbps, hours 0–23):\n"
        "{predicted_values}\n\n"
        "Respond in EXACTLY this two-line format with numeric values only "
        "(no extra text before, between, or after):\n"
        "f_act: A_actual × sin(ω_actual × t + φ_actual) + C_actual\n"
        "f_pred: A_pred × sin(ω_pred × t + φ_pred) + C_pred\n\n"
        "Example:\n"
        "f_act: 210.34 * sin(0.2618 * t - 1.5708) + 623.45\n"
        "f_pred: 195.72 * sin(0.2618 * t - 1.4320) + 605.88"
    ),
)
