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
