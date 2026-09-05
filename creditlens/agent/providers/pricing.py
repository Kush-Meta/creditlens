"""Token pricing, per provider and model.

Only rates that can be stated with confidence are shipped. Everything else
must be configured, and an unconfigured model reports
`cost_basis: "no rate configured"` rather than $0.00 - in a system whose entire
premise is refusing to state numbers it cannot support, silently inventing a
cost for an unpriced model would be self-defeating.

Configure additional rates with `CREDITLENS_MODEL_PRICES`, a comma-separated
list of `model=input_per_mtok/output_per_mtok`:

    CREDITLENS_MODEL_PRICES="gpt-5=1.25/10,llama-3.3-70b=0.6/0.6"
"""
from __future__ import annotations

import os

#: USD per million tokens, (input, output). Anthropic rates as published.
PUBLISHED: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

#: Locally hosted models cost no tokens. Matched by provider, not model name.
FREE_PROVIDERS = {"ollama", "offline"}


def _configured() -> dict[str, tuple[float, float]]:
    raw = os.environ.get("CREDITLENS_MODEL_PRICES", "").strip()
    out: dict[str, tuple[float, float]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        model, _, rates = entry.partition("=")
        try:
            price_in, _, price_out = rates.partition("/")
            out[model.strip()] = (float(price_in), float(price_out))
        except ValueError:
            continue
    return out


def price_for(model: str, provider: str = "") -> tuple[float, float] | None:
    """Rate for a model, or None when no rate is known."""
    if provider in FREE_PROVIDERS:
        return (0.0, 0.0)
    configured = _configured()
    if model in configured:
        return configured[model]
    if model in PUBLISHED:
        return PUBLISHED[model]
    # tolerate vendor prefixes such as "anthropic.claude-opus-5" on Bedrock
    for known, rate in PUBLISHED.items():
        if model.endswith(known):
            return rate
    return None


def known_models() -> list[str]:
    return sorted(set(PUBLISHED) | set(_configured()))
