"""
LangGraph shared state definition for the LLM Network Traffic Prediction pipeline.

This TypedDict is passed through every node in the graph. Each node reads
required fields and returns a dict with only the fields it modifies — LangGraph
merges the updates back into the full state automatically.
"""

from typing import TypedDict, List


class TrafficPredictionState(TypedDict):
    # ── Input data ────────────────────────────────────────────────────────────
    x_t: List[float]             # Input hourly traffic for the previous day (24 values)
    ground_truth: List[float]    # Actual traffic on the target date (24 values, for eval)
    target_date: str             # ISO date string being predicted (e.g. "2024-01-15")

    # ── Prompt components ─────────────────────────────────────────────────────
    p_exam: str                  # Few-shot example block injected into the initial prompt
    p_input: str                 # Input-data block (formatted x_t)
    p_ques: str                  # Question / task instruction block

    # ── Prediction history ────────────────────────────────────────────────────
    y_hat_current: List[float]        # Current best prediction ŷᵢ (24 values)
    y_hat_history: List[List[float]]  # All predictions across iterations [ŷ₀, ŷ₁, …]

    # ── Feedback components ───────────────────────────────────────────────────
    mae_score: float             # MAE for the current prediction
    mae_history: List[float]     # MAE at every evaluation step [MAE₀, MAE₁, …]
    sine_fit_actual: str         # f_act — sine/cosine string for the ground truth
    sine_fit_predicted: str      # f_pred — sine/cosine string for the current prediction
    pfeed_current: str           # Assembled feedback block for the current iteration
    pfeed_history: List[str]     # All previous feedback blocks

    # ── Refinement ────────────────────────────────────────────────────────────
    prefine_current: str         # Refinement instruction built from MAE trend analysis

    # ── Control flow ──────────────────────────────────────────────────────────
    iteration: int               # Current refinement iteration index (incremented in refine node)
    max_iterations: int          # Hard upper bound on refinement iterations
    convergence_threshold: float # MAE Δ below which the pipeline is considered converged
    converged: bool              # True once a stop condition has been met

    # ── Telemetry ─────────────────────────────────────────────────────────────
    api_calls_count: int         # Running total of LLM API calls made
    total_tokens_used: int       # Running total of tokens consumed across all calls
