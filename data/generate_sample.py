"""
CLI script — generate a sample traffic_data.csv for local testing.

Usage:
    python data/generate_sample.py                        # 30-day dataset
    python data/generate_sample.py --days 60 --out path  # custom
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate_csv(days: int = 30, seed: int = 42, out_path: str = "data/traffic_data.csv") -> str:
    """
    Write a synthetic hourly traffic CSV with realistic daily patterns.

    Columns: datetime (ISO-8601), traffic_volume (Mbps float).

    Args:
        days:     Number of days to generate.
        seed:     NumPy random seed.
        out_path: Destination CSV path.

    Returns:
        Absolute path of the written file.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2024-01-01")
    timestamps = pd.date_range(start, periods=days * 24, freq="h")

    hours = np.tile(np.arange(24, dtype=float), days)
    day_idx = np.repeat(np.arange(days), 24)

    # Weekly pattern: weekends slightly lower
    weekday = timestamps.dayofweek.to_numpy()   # 0=Mon … 6=Sun
    is_weekend = (weekday >= 5).astype(float)
    weekend_factor = 1.0 - 0.15 * is_weekend   # −15 % on weekends

    # Gradual upward trend across the dataset
    trend = 1.0 + 0.002 * day_idx

    # Intra-day shape
    base      = 400.0
    daily     = 200.0 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    morn_peak = 180.0 * np.exp(-0.5 * ((hours - 8.5) / 1.5) ** 2)
    eve_peak  = 160.0 * np.exp(-0.5 * ((hours - 17.5) / 1.5) ** 2)
    noise     = rng.normal(0, 28, len(timestamps))

    traffic = (base + daily + morn_peak + eve_peak) * weekend_factor * trend + noise
    traffic = np.clip(traffic, 50.0, None)

    df = pd.DataFrame({"datetime": timestamps, "traffic_volume": traffic.round(2)})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[generate_sample] Wrote {len(df):,} rows → {os.path.abspath(out_path)}")
    print(f"  date range  : {timestamps[0].date()}  →  {timestamps[-1].date()}")
    print(f"  traffic mean: {traffic.mean():.1f} Mbps   std: {traffic.std():.1f} Mbps")
    return os.path.abspath(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic traffic CSV for testing.")
    parser.add_argument("--days", type=int, default=30, help="Number of days to generate")
    parser.add_argument("--seed", type=int, default=42, help="NumPy random seed")
    parser.add_argument(
        "--out", type=str, default="data/traffic_data.csv", help="Output CSV path"
    )
    args = parser.parse_args()
    generate_csv(days=args.days, seed=args.seed, out_path=args.out)
