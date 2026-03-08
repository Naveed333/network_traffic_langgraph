"""
Structured per-node logging utilities.

All graph nodes call log_node_entry() on entry and log_node_result() on exit
so that token counts, iteration numbers, and parsed outputs appear uniformly
in the run log.
"""

import logging
import sys
from typing import Any, Optional


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that emits to stdout with a consistent format.

    Idempotent — calling twice on the same name returns the same logger
    without duplicating handlers.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger


# ── Per-node helpers ──────────────────────────────────────────────────────────

def log_node_entry(
    logger: logging.Logger,
    node_name: str,
    iteration: int,
    **extra: Any,
) -> None:
    """
    Emit a standardised entry banner for a graph node.

    Args:
        logger:    The node's logger.
        node_name: Canonical node name (matches graph.add_node key).
        iteration: Current pipeline iteration counter from state.
        **extra:   Optional key=value pairs appended to the log line.
    """
    parts = [f"[ENTER] node={node_name!r:<30} iter={iteration}"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))


def log_node_result(
    logger: logging.Logger,
    node_name: str,
    iteration: int,
    token_count: Optional[int] = None,
    parsed_output: Optional[Any] = None,
    **extra: Any,
) -> None:
    """
    Emit a standardised exit summary for a graph node.

    Args:
        logger:        The node's logger.
        node_name:     Canonical node name.
        iteration:     Current pipeline iteration counter.
        token_count:   Number of tokens consumed by the LLM call (if applicable).
        parsed_output: The value parsed from the LLM response (truncated for display).
        **extra:       Optional additional key=value pairs.
    """
    output_str: str = ""
    if parsed_output is not None:
        raw = str(parsed_output)
        output_str = (raw[:120] + "…") if len(raw) > 120 else raw

    parts = [f"[EXIT ] node={node_name!r:<30} iter={iteration}"]
    if token_count is not None:
        parts.append(f"tokens={token_count:,}")
    if output_str:
        parts.append(f"output={output_str!r}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")

    logger.info(" | ".join(parts))
