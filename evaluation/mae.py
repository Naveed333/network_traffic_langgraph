"""
Local Mean Absolute Error (MAE) computation.

Used as the ground-truth fallback when the LLM-based MAE node fails to
return a parseable numeric value.
"""

from typing import List
import numpy as np


def compute_mae(actual: List[float], predicted: List[float]) -> float:
    """
    Compute the Mean Absolute Error between actual and predicted traffic values.

    Args:
        actual:    Ground-truth hourly traffic (24 values).
        predicted: Predicted hourly traffic (24 values).

    Returns:
        MAE as a non-negative float.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )

    actual_arr = np.array(actual, dtype=float)
    predicted_arr = np.array(predicted, dtype=float)
    return float(np.mean(np.abs(actual_arr - predicted_arr)))


def compute_rmse(actual: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error — supplementary metric for final reporting."""
    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )
    actual_arr = np.array(actual, dtype=float)
    predicted_arr = np.array(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))


def compute_mape(actual: List[float], predicted: List[float]) -> float:
    """Mean Absolute Percentage Error — avoids division-by-zero with small epsilon."""
    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )
    actual_arr = np.array(actual, dtype=float)
    predicted_arr = np.array(predicted, dtype=float)
    eps = 1e-8
    return float(np.mean(np.abs((actual_arr - predicted_arr) / (actual_arr + eps))) * 100)
