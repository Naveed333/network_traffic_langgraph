"""
LLM response parsers.

Every node that calls an LLM delegates its output parsing to one of the
functions below.  Each parser is lenient — it uses multiple extraction
strategies and returns None on total failure so the caller can retry.
"""

import json
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regex that matches any numeric token (int or float, incl. scientific notation)
_NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


# ── 24-value prediction parser ────────────────────────────────────────────────

def parse_24_floats(response: str) -> Optional[List[float]]:
    """
    Extract exactly 24 comma-separated float values from an LLM response.

    Strategy 1 — find a line that splits into exactly 24 numeric tokens.
    Strategy 2 — extract all numbers from the response and take the first 24.

    Returns:
        List of 24 floats, or None if parsing fails entirely.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```[^`]*```", " ", response, flags=re.DOTALL)
    cleaned = cleaned.replace("`", "").strip()

    # Strategy 1: look for a line with exactly 24 comma-separated values
    for line in cleaned.splitlines():
        line = line.strip().strip(",").strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 24:
            try:
                values = [float(p) for p in parts]
                logger.debug("parse_24_floats: strategy-1 succeeded")
                return values
            except ValueError:
                continue

    # Strategy 2: extract all numeric tokens and grab first 24
    numbers = _NUMBER_RE.findall(cleaned)
    if len(numbers) >= 24:
        try:
            values = [float(n) for n in numbers[:24]]
            logger.warning(
                "parse_24_floats: strategy-2 fallback used (%d numbers found)", len(numbers)
            )
            return values
        except ValueError:
            pass

    logger.error(
        "parse_24_floats: failed. Response snippet: %r", response[:300]
    )
    return None


# ── Single float parser ───────────────────────────────────────────────────────

def parse_float(response: str) -> Optional[float]:
    """
    Extract the first numeric value from an LLM response.

    Handles integers, decimals, and scientific notation.

    Returns:
        Float value, or None if no number can be extracted.
    """
    numbers = _NUMBER_RE.findall(response.strip())
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass

    logger.error("parse_float: failed. Response snippet: %r", response[:200])
    return None


# ── Sine-formula pair parser ──────────────────────────────────────────────────

def parse_sine_strings(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract f_act and f_pred sine-formula strings from an LLM response.

    Parsing strategy (in priority order):
      1. JSON — SINE_FIT_TEMPLATE now requests {"f_act": "...", "f_pred": "..."}
      2. Explicit label lines — "f_act: <formula>" / "f_pred: <formula>"
      3. Heuristic — first two lines that contain 'sin' or 'cos'

    Returns:
        (f_act, f_pred) — either or both may be None if extraction fails.
    """
    f_act:  Optional[str] = None
    f_pred: Optional[str] = None

    # ── Strategy 1: JSON parse ────────────────────────────────────────────────
    # Strip markdown fences in case the model wrapped the JSON in ```json ... ```
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", response).replace("```", "").strip()
    # Find the first {...} block
    json_match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            f_act  = data.get("f_act")  or data.get("f_actual")  or data.get("actual")
            f_pred = data.get("f_pred") or data.get("f_predicted") or data.get("predicted")
            if f_act and f_pred:
                logger.debug("parse_sine_strings: JSON strategy succeeded")
                return f_act, f_pred
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Strategy 2: explicit label lines ─────────────────────────────────────
    for line in response.splitlines():
        line_lower = line.lower().strip()

        if "f_act" in line_lower or ("actual" in line_lower and re.search(r"sin|cos", line_lower)):
            m = re.search(r"[:=]\s*(.+)", line)
            if m:
                f_act = m.group(1).strip().strip('"').strip("'")

        elif "f_pred" in line_lower or ("predicted" in line_lower and re.search(r"sin|cos", line_lower)):
            m = re.search(r"[:=]\s*(.+)", line)
            if m:
                f_pred = m.group(1).strip().strip('"').strip("'")

    if f_act and f_pred:
        logger.debug("parse_sine_strings: label strategy succeeded")
        return f_act, f_pred

    # ── Strategy 3: heuristic — grab first two sin/cos lines ─────────────────
    formula_lines = [
        ln.strip()
        for ln in response.splitlines()
        if re.search(r"\b(sin|cos|sine|cosine)\b", ln, re.IGNORECASE)
    ]
    if len(formula_lines) >= 2:
        if not f_act:
            f_act = formula_lines[0]
        if not f_pred:
            f_pred = formula_lines[1]
    elif len(formula_lines) == 1:
        if not f_act:
            f_act = formula_lines[0]
        if not f_pred:
            f_pred = formula_lines[0]

    if not f_act or not f_pred:
        logger.warning(
            "parse_sine_strings: partial parse — f_act=%s f_pred=%s",
            f_act is not None, f_pred is not None,
        )

    return f_act, f_pred
