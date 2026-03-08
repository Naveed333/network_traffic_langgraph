"""
tiktoken-based token counting and context-window budget enforcement.

The context guard runs BEFORE every LLM call inside the refine_predict node.
If the assembled prompt exceeds 80 % of the model's context window, the node
must summarise its history before proceeding.
"""

import logging
from typing import List, Tuple

import tiktoken

logger = logging.getLogger(__name__)

# Models that use the newer o200k_base encoding (GPT-4o family)
_O200K_MODELS = {"gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"}


# ── Encoding helper ────────────────────────────────────────────────────────────

def _get_encoding(model: str) -> tiktoken.Encoding:
    """
    Return the tiktoken encoding for *model*, falling back to cl100k_base
    for any model not yet registered in tiktoken's registry.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # GPT-4o uses o200k_base; unknown future models default to cl100k_base
        if any(m in model for m in _O200K_MODELS):
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")


# ── Public API ────────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in *text* using tiktoken.

    Args:
        text:  Plain text to tokenise.
        model: OpenAI model name used to select the correct encoding.

    Returns:
        Integer token count.
    """
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


def count_messages_tokens(messages: List[dict], model: str = "gpt-4o") -> int:
    """
    Count tokens for a list of OpenAI-format chat messages.

    Applies the per-message overhead (4 tokens) and priming overhead (2 tokens)
    that the API itself charges.

    Args:
        messages: List of {"role": str, "content": str} dicts.
        model:    Model name for encoding selection.

    Returns:
        Total token count including overhead.
    """
    encoding = _get_encoding(model)
    total = 2  # priming
    for msg in messages:
        total += 4  # per-message overhead
        for value in msg.values():
            total += len(encoding.encode(str(value)))
    return total


def check_context_guard(
    text: str,
    context_limit: int,
    guard_pct: float = 0.8,
    model: str = "gpt-4o",
) -> Tuple[bool, int]:
    """
    Determine whether *text* fits within the allowed token budget.

    Args:
        text:          The assembled prompt text to evaluate.
        context_limit: Total context window size (e.g. 128_000 for GPT-4o).
        guard_pct:     Fraction of context_limit that triggers a warning (default 0.8).
        model:         Model name for encoding selection.

    Returns:
        (within_budget, token_count) where *within_budget* is True iff
        token_count <= floor(context_limit * guard_pct).
    """
    token_count = count_tokens(text, model)
    budget = int(context_limit * guard_pct)
    within_budget = token_count <= budget

    if not within_budget:
        logger.warning(
            "Context guard triggered: %d tokens > %d budget (%.0f%% of %d)",
            token_count, budget, guard_pct * 100, context_limit,
        )
    else:
        logger.debug(
            "Context guard OK: %d / %d tokens (%.1f%% used)",
            token_count, budget, token_count / budget * 100,
        )

    return within_budget, token_count
