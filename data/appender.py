"""
data/appender.py — Append new daily traffic data to the CSV.

Run every morning once yesterday's real data is collected:

    # From comma-separated values:
    python data/appender.py \\
        --date 2024-07-14 \\
        --values "47.2,58.1,45.3,36.8,54.2,57.9,58.6,69.4,64.1,77.3,93.8,75.4,
                  61.1,63.9,87.7,56.9,63.2,60.1,66.8,66.1,61.7,50.3,45.2,47.1"

    # From a source CSV (must have datetime + traffic_volume columns):
    python data/appender.py \\
        --date 2024-07-14 \\
        --source new_data.csv

After appending, the next context pipeline run can use the new day as input
or as a ground-truth evaluation target.
"""

import argparse
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CSV = os.path.join(
    os.path.dirname(__file__), "traffic_data.csv"
)


def append_daily_data(
    filepath: str,
    date: str,
    hourly_values: list,
) -> None:
    """
    Append 24 hourly rows for *date* to the traffic CSV.

    Args:
        filepath:      Path to the existing traffic CSV.
        date:          ISO date string (e.g. "2024-07-14").
        hourly_values: Exactly 24 float values (Mbps), one per hour.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError:        If date already present, or values count != 24,
                           or any value is negative.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV not found: {filepath}")

    if len(hourly_values) != 24:
        raise ValueError(
            f"Expected exactly 24 hourly values, got {len(hourly_values)}."
        )

    if any(v < 0 for v in hourly_values):
        raise ValueError(
            "All traffic values must be non-negative (Mbps). "
            f"Got negatives: {[v for v in hourly_values if v < 0]}"
        )

    # ── Load existing CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    existing_dates = set(df["datetime"].dt.date.unique())

    target = pd.Timestamp(date).date()
    if target in existing_dates:
        raise ValueError(
            f"Date {date} already exists in {filepath}. "
            "Remove it first or choose a different date."
        )

    # ── Build 24 new rows ──────────────────────────────────────────────────────
    new_rows = []
    last_id  = int(df["id_time"].max()) if "id_time" in df.columns else len(df) - 1

    for hour, value in enumerate(hourly_values):
        ts = pd.Timestamp(f"{date} {hour:02d}:00:00+00:00")
        row = {col: None for col in df.columns}
        row["datetime"]       = ts
        row["traffic_volume"] = float(value)
        if "id_time" in df.columns:
            row["id_time"] = last_id + 1 + hour
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    # ── Append and save ────────────────────────────────────────────────────────
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)
    combined.to_csv(filepath, index=False)

    date_range_start = combined["datetime"].min().date()
    date_range_end   = combined["datetime"].max().date()

    logger.info(
        "Appended %d rows for %s → %s | CSV now spans %s to %s (%d total rows)",
        len(new_rows), date, filepath,
        date_range_start, date_range_end, len(combined),
    )


def load_from_source_csv(source_path: str, date: str) -> list:
    """
    Extract 24 hourly traffic_volume values for *date* from a source CSV.

    Args:
        source_path: Path to source CSV (must have datetime + traffic_volume).
        date:        ISO date string to extract.

    Returns:
        List of 24 float values.
    """
    df = pd.read_csv(source_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    target = pd.Timestamp(date).date()
    rows = df[df["datetime"].dt.date == target]["traffic_volume"].tolist()
    if not rows:
        raise ValueError(
            f"No rows found for date {date} in {source_path}."
        )
    if len(rows) != 24:
        logger.warning(
            "Expected 24 rows for %s in source CSV, found %d — "
            "zero-padding to 24.",
            date, len(rows),
        )
        rows = rows[:24] + [0.0] * (24 - len(rows))
    return [float(v) for v in rows]


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append new daily traffic data to the CSV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="ISO date of the new data (e.g. 2024-07-14).",
    )
    parser.add_argument(
        "--values", type=str, default=None,
        help="24 comma-separated Mbps values for hours 00–23.",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to a source CSV with datetime + traffic_volume columns.",
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help=f"Target CSV to append to (default: {DEFAULT_CSV}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.values and not args.source:
        logger.error("Provide either --values or --source.")
        sys.exit(1)

    if args.values:
        try:
            values = [float(v.strip()) for v in args.values.split(",")]
        except ValueError as exc:
            logger.error("Could not parse --values: %s", exc)
            sys.exit(1)
    else:
        try:
            values = load_from_source_csv(args.source, args.date)
        except Exception as exc:
            logger.error("Failed to load from source CSV: %s", exc)
            sys.exit(1)

    try:
        append_daily_data(args.csv, args.date, values)
    except Exception as exc:
        logger.error("Append failed: %s", exc)
        sys.exit(1)
