"""Provider-neutral conversation model and engine interface.

The orchestrator must not know which vendor is answering. That means the loop
cannot pass vendor-shaped message dicts around, because the three major tool
protocols disagree in ways that cannot be papered over at the edges:

* **Anthropic** puts `tool_use` blocks inside an assistant message's content and
  expects `tool_result` blocks inside a following *user* message.
* **OpenAI** puts `tool_calls` on the assistant message and expects one message
  per result with `role="tool"` and a `tool_call_id`.
* **Google** uses `functionCall` / `functionResponse` parts, with results in a
  message whose role is `"user"` but whose parts are function responses.

So the loop speaks in `Turn` objects - a user message, an assistant turn, or a
batch of tool results - and each engine translates that transcript into its own
wire format on every call. Engines are stateless with respect to the
conversation: they are handed the whole transcript each time, which keeps
retries, engine substitution mid-run, and replay all trivially correct.

`AssistantTurn` carries `raw` - the provider's own representation of that turn.
Replaying it verbatim matters for providers that require it (Anthropic thinking
blocks must be echoed back unchanged), and is ignored by providers that do not.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# usage and cost
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.requests += other.requests

    def cost_usd(self, model: str, provider: str = "") -> float:
        from creditlens.agent.providers.pricing import price_for

        rate = price_for(model, provider)
        if rate is None:
            return 0.0
        price_in, price_out = rate
        return (
            self.input_tokens * price_in
            + self.cache_read_tokens * price_in * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * price_in * CACHE_WRITE_MULTIPLIER
            + self.output_tokens * price_out
        ) / 1_000_000

    def to_dict(self, model: str = "", provider: str = "") -> dict[str, Any]:
        from creditlens.agent.providers.pricing import price_for

        priced = price_for(model, provider) is not None
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "estimated_cost_usd": round(self.cost_usd(model, provider), 6) if model else None,
            # Never let an unpriced model silently report $0.00 as if it were free.
            "cost_basis": "published rates" if priced else "no rate configured",
        }


CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


# ---------------------------------------------------------------------------
# conversation
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str  # JSON-encoded tool output
    is_error: bool = False


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    #: the provider's own representation of this turn, replayed verbatim when
    #: the provider needs it (Anthropic thinking blocks) and ignored otherwise
    raw: Any = None
    thinking: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class UserMessage:
    text: str


@dataclass
class AssistantTurn:
    response: LLMResponse


@dataclass
class ToolResultBatch:
    results: list[ToolResult]


Turn = UserMessage | AssistantTurn | ToolResultBatch
Conversation = list[Turn]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Recoverable provider failure."""


class LLMUnavailable(LLMError):
    """No usable credentials, no connectivity, or SDK not installed."""


class LLMRefusal(LLMError):
    """The provider declined to answer."""


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class LLMEngine(ABC):
    #: stable identifier reported on every analysis, so an answer can always be
    #: attributed to the engine that produced it
    name: str
    provider: str = ""
    model: str = ""

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        conversation: Conversation,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        force_tool: str | None = None,
    ) -> LLMResponse:
        """Answer the transcript, optionally forcing one specific tool call."""

    @property
    def supports_tools(self) -> bool:
        return True

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "provider": self.provider, "model": self.model}


# ---------------------------------------------------------------------------
# canonical tool schema helpers
# ---------------------------------------------------------------------------
def tool_schemas_to_openai(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical schema -> OpenAI `function` tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def tool_schemas_to_google(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical schema -> Google function declarations.

    Google's schema dialect rejects several JSON Schema keywords, so the
    parameter schema is sanitised rather than passed through.
    """
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": _sanitize_google_schema(
                tool.get("input_schema", {"type": "object", "properties": {}})
            ),
        }
        for tool in tools
    ]


_GOOGLE_UNSUPPORTED = {"additionalProperties", "$schema", "minItems", "maxItems", "default"}


def _sanitize_google_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            key: _sanitize_google_schema(value)
            for key, value in schema.items()
            if key not in _GOOGLE_UNSUPPORTED
        }
    if isinstance(schema, list):
        return [_sanitize_google_schema(item) for item in schema]
    return schema
