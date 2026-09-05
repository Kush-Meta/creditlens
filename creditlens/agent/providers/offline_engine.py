"""The deterministic offline engine.

Not a mock that returns canned strings: it runs the same tool loop, calls the
same deterministic finance code, and writes its narrative from the tool
results. That keeps retrieval, calculation, verification and evaluation fully
testable with no network and no credentials, gives the service somewhere to
degrade to when a provider is unreachable, and serves as an eval baseline -
the gap between it and a model measures what the model actually contributes.

It is the only engine that is always available, which is why it is also the
universal fallback for every other provider.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from creditlens.agent.providers.base import Conversation, LLMEngine, LLMResponse

_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("compare", re.compile(r"\b(compare|versus|vs\.?|against|which of|stronger)\b", re.I)),
    ("trend", re.compile(
        r"\b(trend|over the (last|past)|evolved|improv|deteriorat|history|four quarters|three years)\b",
        re.I)),
    ("cause", re.compile(r"\b(why|what caused|driver|drove|explain the change|change in)\b", re.I)),
    ("risk", re.compile(r"\b(risk|threat|headwind|concern|downside|argument against)\b", re.I)),
    ("ratio", re.compile(
        r"\b(calculate|compute|ratio|leverage|coverage|liquidity|margin|debt.to|ebitda)\b", re.I)),
    ("profile", re.compile(
        r"\b(credit profile|creditworth|summar|overall|scorecard|quality)\b", re.I)),
)


class OfflineEngine(LLMEngine):
    """Deterministic planner and synthesizer. No network, no key, no randomness."""

    name = "offline-deterministic"
    provider = "offline"

    def __init__(self, model: str = "offline-deterministic-v1"):
        self.model = model

    def complete(
        self,
        *,
        system: str,
        conversation: Conversation,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        force_tool: str | None = None,
    ) -> LLMResponse:
        from creditlens.agent.offline_planner import plan_next_step

        return plan_next_step(conversation=conversation, tools=tools or [])


def classify_intents(question: str) -> list[str]:
    return [name for name, pattern in _INTENT_PATTERNS if pattern.search(question)] or ["profile"]
