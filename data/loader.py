"""
Traffic data loading and synthetic data generation.

Supports:
  - Loading from a CSV file with columns [datetime, traffic_volume].
  - Falling back to in-memory synthetic data when no CSV is available.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_traffic_data(
    filepath: str,
    target_date: Optional[str] = None,
) -> Tuple[List[float], List[float], str]:
    """
    Load 24-hour traffic sequences from a CSV file.

    The CSV must contain at least two columns:
        - ``datetime``: ISO-8601 timestamp strings (hourly resolution).
        - ``traffic_volume``: Numeric traffic (Mbps or packet counts).

    Args:
        filepath:    Path to the CSV file.
        target_date: ISO date string (``"YYYY-MM-DD"``) for the day to predict.
                     When omitted the function uses the last two available dates:
                     second-to-last as *x_t* input and last as *ground_truth*.

    Returns:
        (x_t, ground_truth, target_date_str) — three-tuple of 24-value lists
        and the resolved target date string.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError:        If fewer than two days of data are present.
    """
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    unique_dates = sorted(df["datetime"].dt.date.unique())

    if len(unique_dates) < 2:
        raise ValueError(
            f"Need at least 2 days of data in {filepath}; found {len(unique_dates)}."
        )

    if target_date:
        tgt = pd.Timestamp(target_date).date()
        prev = tgt - pd.Timedelta(days=1)
        if tgt not in unique_dates or prev not in unique_dates:
            raise ValueError(
                f"target_date={target_date} or previous day not found in {filepath}."
            )
    else:
        tgt  = unique_dates[-1]
        prev = unique_dates[-2]

    def _extract_day(date) -> List[float]:
        rows = df[df["datetime"].dt.date == date]["traffic_volume"].tolist()
        rows = rows[:24]                          # cap at 24 hours
        rows = rows + [0.0] * (24 - len(rows))   # zero-pad if incomplete
        return [float(v) for v in rows]

    x_t          = _extract_day(prev)
    ground_truth = _extract_day(tgt)
    target_str   = str(tgt)

    logger.info(
        "Loaded data: prev_date=%s  target_date=%s  "
        "x_t=[%.1f…%.1f]  gt=[%.1f…%.1f]",
        prev, target_str,
        x_t[0], x_t[-1],
        ground_truth[0], ground_truth[-1],
    )
    return x_t, ground_truth, target_str


# ── Context-pipeline helpers ──────────────────────────────────────────────────

def validate_context_dates(
    filepath: str,
    target_date: str,
    context_days: int,
) -> None:
    """
    Verify that all dates required for the context pipeline exist in the CSV.

    For context_days=4 and target=Jul 14, the following must be present:
        Jul 9  (input for eval Jul 10)
        Jul 10 (ground truth for eval Jul 10, input for eval Jul 11)
        Jul 11 (ground truth for eval Jul 11, input for eval Jul 12)
        Jul 12 (ground truth for eval Jul 12, input for eval Jul 13)
        Jul 13 (ground truth for eval Jul 13, input for deployment)

    Args:
        filepath:     Path to CSV file.
        target_date:  ISO date string of the day to predict (must NOT be in CSV).
        context_days: Number of previous days to evaluate.

    Raises:
        ValueError: If any required date is missing from the CSV.
    """
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    unique_dates = set(df["datetime"].dt.date.unique())

    tgt  = pd.Timestamp(target_date).date()
    # Need context_days evaluation targets + their previous days
    # e.g. context_days=4, target=Jul14 → need Jul9..Jul13
    required = [
        tgt - pd.Timedelta(days=i)
        for i in range(1, context_days + 2)   # +2 to include the extra prev-day
    ]
    missing = [str(d) for d in required if d not in unique_dates]
    if missing:
        raise ValueError(
            f"Context pipeline requires {context_days + 1} consecutive days before "
            f"{target_date}. Missing from {filepath}: {missing}"
        )
    if tgt in unique_dates:
        raise ValueError(
            f"target_date={target_date} already exists in {filepath}. "
            "Cannot predict a date that is already in the dataset."
        )
    logger.info(
        "validate_context_dates: all required dates present for context_days=%d, "
        "target=%s",
        context_days, target_date,
    )


def load_deployment_input(
    filepath: str,
    target_date: str,
) -> Tuple[List[float], str]:
    """
    Load x[t] for deployment — the day immediately before target_date.

    target_date must NOT exist in the CSV (it is the future day to predict).

    Args:
        filepath:    Path to CSV file.
        target_date: ISO date string of the day to predict.

    Returns:
        (x_t, target_date_str) — 24-value input list and target date string.
    """
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    tgt  = pd.Timestamp(target_date).date()
    prev = tgt - pd.Timedelta(days=1)

    unique_dates = set(df["datetime"].dt.date.unique())
    if prev not in unique_dates:
        raise ValueError(
            f"Cannot load deployment input: {prev} not found in {filepath}."
        )

    def _extract_day(date) -> List[float]:
        rows = df[df["datetime"].dt.date == date]["traffic_volume"].tolist()
        rows = rows[:24]
        rows = rows + [0.0] * (24 - len(rows))
        return [float(v) for v in rows]

    x_t = _extract_day(prev)
    logger.info(
        "load_deployment_input: x_t=%s  target=%s", str(prev), str(tgt)
    )
    return x_t, str(tgt)


# ── Synthetic data ────────────────────────────────────────────────────────────

def generate_synthetic_data(seed: int = 42) -> Tuple[List[float], List[float], str]:
    """
    Generate two days of realistic synthetic hourly traffic with:
        - Low overnight baseline (00–05 h).
        - Morning peak  (~08–09 h).
        - Midday plateau (~11–13 h).
        - Evening peak   (~17–18 h).
        - Additive Gaussian noise.

    Args:
        seed: NumPy random seed for reproducibility.

    Returns:
        (x_t, ground_truth, target_date) — same shape as ``load_traffic_data``.
    """
    rng   = np.random.default_rng(seed)
    hours = np.arange(24, dtype=float)

    def _traffic_pattern(scale: float = 1.0) -> np.ndarray:
        base      = 400.0                                        # overnight baseline
        daily     = 200.0 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
        morn_peak = 180.0 * np.exp(-0.5 * ((hours - 8.5) / 1.5) ** 2)
        eve_peak  = 160.0 * np.exp(-0.5 * ((hours - 17.5) / 1.5) ** 2)
        noise     = rng.normal(0, 25, 24)
        pattern   = scale * (base + daily + morn_peak + eve_peak) + noise
        return np.clip(pattern, 50.0, None)          # floor at 50 Mbps

    x_t_arr      = _traffic_pattern(scale=1.00)
    gt_arr       = _traffic_pattern(scale=1.04)   # next day ~4 % higher

    x_t          = x_t_arr.tolist()
    ground_truth = gt_arr.tolist()
    target_date  = "2024-01-02"

    logger.info(
        "Generated synthetic data: x_t_mean=%.1f  gt_mean=%.1f",
        float(np.mean(x_t_arr)),
        float(np.mean(gt_arr)),
    )
    return x_t, ground_truth, target_date
