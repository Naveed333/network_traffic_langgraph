"""
SciPy-based sine/cosine curve fitting for traffic pattern analysis.

Used as a local fallback when the LLM sine-fitting node returns
an unparseable response.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


# ── Model definition ──────────────────────────────────────────────────────────

def sine_model(t: np.ndarray, A: float, omega: float, phi: float, C: float) -> np.ndarray:
    """
    Parametric sine model: f(t) = A * sin(omega * t + phi) + C

    Args:
        t:     Time indices (0–23 for a 24-hour window).
        A:     Amplitude — half the peak-to-trough range.
        omega: Angular frequency (2π/24 ≈ 0.2618 for a daily cycle).
        phi:   Phase shift in radians.
        C:     Vertical offset (mean traffic level).
    """
    return A * np.sin(omega * t + phi) + C


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_sine(values: List[float]) -> Tuple[float, float, float, float]:
    """
    Fit a sine curve to a 24-point traffic sequence via non-linear least squares.

    Returns:
        Tuple (A, omega, phi, C) of best-fit parameters.
        Falls back to sensible defaults if scipy fails.
    """
    t = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)

    C_guess     = float(np.mean(y))
    omega_guess = 2.0 * np.pi / 24.0   # one full daily cycle
    phi_guess   = -np.pi / 2.0         # peak near midday

    # ── Robust bounds — safe even when all values are zero ────────────────────
    # y_abs_max  >= 1.0  ensures upper > lower for A and C bounds at all times.
    # y_range    >= 1.0  ensures A_guess is always positive.
    y_abs_max = max(float(np.abs(y).max()), 1.0)
    y_range   = max(float(y.max() - y.min()), 1.0)
    A_guess   = y_range / 2.0

    try:
        popt, _ = curve_fit(
            sine_model,
            t,
            y,
            p0=[A_guess, omega_guess, phi_guess, C_guess],
            maxfev=10_000,
            bounds=(
                # lower:  A >= 0,  ω > 0,  φ ∈ [-2π, 2π],  C unconstrained below
                [0.0,            0.001,   -2.0 * np.pi,  -y_abs_max * 2.0],
                # upper:  always strictly > lower because y_abs_max >= 1.0
                [y_abs_max * 3,  np.pi,    2.0 * np.pi,   y_abs_max * 3.0],
            ),
        )
        return tuple(float(p) for p in popt)  # type: ignore[return-value]

    except Exception as exc:
        logger.warning("scipy curve_fit failed: %s — returning initial guess", exc)
        return A_guess, omega_guess, phi_guess, C_guess


# ── Formatting ────────────────────────────────────────────────────────────────

def format_sine_string(A: float, omega: float, phi: float, C: float) -> str:
    """
    Render sine parameters as a human-readable formula string.

    Example: "183.45 * sin(0.2618 * t - 1.5708) + 612.30"
    """
    phi_op = "+" if phi >= 0 else "-"
    C_op   = "+" if C  >= 0 else "-"
    return (
        f"{A:.4f} * sin({omega:.4f} * t {phi_op} {abs(phi):.4f})"
        f" {C_op} {abs(C):.4f}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_sine_fits(
    actual: List[float],
    predicted: List[float],
) -> Tuple[str, str]:
    """
    Fit sine curves to both actual and predicted traffic sequences.

    Returns:
        (f_act, f_pred) — formatted sine-formula strings.
    """
    A_a, w_a, p_a, C_a = fit_sine(actual)
    A_p, w_p, p_p, C_p = fit_sine(predicted)

    f_act  = format_sine_string(A_a, w_a, p_a, C_a)
    f_pred = format_sine_string(A_p, w_p, p_p, C_p)

    logger.debug("Sine fit — actual : %s", f_act)
    logger.debug("Sine fit — predict: %s", f_pred)

    return f_act, f_pred
