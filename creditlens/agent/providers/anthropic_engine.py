"""Anthropic provider.

Also covers Anthropic on Amazon Bedrock, Google Vertex AI and Microsoft
Foundry, which expose the same Messages surface behind a different client
class, so only construction differs.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from creditlens.agent.providers.base import (
    AssistantTurn,
    Conversation,
    LLMEngine,
    LLMError,
    LLMRefusal,
    LLMResponse,
    LLMUnavailable,
    ToolCall,
    ToolResultBatch,
    Usage,
    UserMessage,
)
from creditlens.observability import METRICS, get_logger

log = get_logger(__name__)

#: Models on which `budget_tokens` is rejected and adaptive thinking is the
#: only supported configuration.
ADAPTIVE_THINKING_MODELS = (
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5", "claude-mythos-5",
)


class AnthropicEngine(LLMEngine):
    name = "anthropic"
    provider = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        *,
        backend: str = "anthropic",
        settings: Any | None = None,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMUnavailable("the `anthropic` package is not installed") from exc

        from creditlens.config import get_settings

        self.settings = settings or get_settings()
        self.model = model or self.settings.model
        self.backend = backend
        self._anthropic = anthropic
        self.client = client or self._build_client(anthropic, backend)

    def _build_client(self, anthropic: Any, backend: str) -> Any:
        timeout = self.settings.llm_timeout_s
        retries = self.settings.llm_max_retries
        try:
            if backend == "bedrock":
                return anthropic.AnthropicBedrock(
                    aws_region=self.settings.aws_region, timeout=timeout, max_retries=retries
                )
            if backend == "vertex":
                return anthropic.AnthropicVertex(
                    project_id=self.settings.gcp_project,
                    region=self.settings.gcp_region,
                    timeout=timeout, max_retries=retries,
                )
            return anthropic.Anthropic(timeout=timeout, max_retries=retries)
        except Exception as exc:
            raise LLMUnavailable(f"could not construct the {backend} client: {exc}") from exc

    # -- wire format ------------------------------------------------------
    def _to_messages(self, conversation: Conversation) -> list[dict[str, Any]]:
        """Neutral transcript -> Anthropic messages.

        Assistant turns replay `raw` verbatim when present: thinking blocks must
        be echoed back unchanged on the same model, and reconstructing them from
        text would silently drop them.
        """
        messages: list[dict[str, Any]] = []
        for turn in conversation:
            if isinstance(turn, UserMessage):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                content = turn.response.raw
                if content is None:
                    content = []
                    if turn.response.text:
                        content.append({"type": "text", "text": turn.response.text})
                    for call in turn.response.tool_calls:
                        content.append({
                            "type": "tool_use", "id": call.id,
                            "name": call.name, "input": call.arguments,
                        })
                messages.append({"role": "assistant", "content": content})
            elif isinstance(turn, ToolResultBatch):
                # Every result for one assistant turn goes back in a single user
                # message; splitting them trains the model out of parallel calls.
                messages.append({"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                    for result in turn.results
                ]})
        return messages

    def complete(
        self,
        *,
        system: str,
        conversation: Conversation,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        force_tool: str | None = None,
    ) -> LLMResponse:
        anthropic = self._anthropic

        # `tools` then `system` form the cached prefix; volatile content follows.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": self._to_messages(conversation),
        }
        if any(self.model.endswith(m) for m in ADAPTIVE_THINKING_MODELS):
            kwargs["output_config"] = {"effort": self.settings.effort}
            if self.settings.thinking:
                kwargs["thinking"] = {"type": "adaptive"}
        if tools:
            kwargs["tools"] = list(tools)
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.NotFoundError as exc:
            raise LLMError(f"model or endpoint not found: {exc}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable(f"authentication failed: {exc}") from exc
        except anthropic.RateLimitError as exc:
            retry_after = (
                exc.response.headers.get("retry-after", "unknown") if exc.response else "?"
            )
            raise LLMError(f"rate limited (retry-after={retry_after})") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"api error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"connection error: {exc}") from exc
        except TypeError as exc:
            # The SDK resolves credentials lazily: construction succeeds with no
            # key and the failure only surfaces on the first request.
            raise LLMUnavailable(f"no usable credentials: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        return self._normalize(response)

    def _normalize(self, response: Any) -> LLMResponse:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                # Inputs arrive parsed; never string-match the serialized form.
                calls.append(ToolCall(id=block.id, name=block.name,
                                      arguments=dict(block.input or {})))

        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            requests=1,
        )
        _record(self.provider, self.model, usage)

        stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(
                "model declined the request" + (f" ({category})" if category else "")
            )

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason=stop_reason,
            usage=usage,
            raw=response.content,
            thinking="\n".join(thinking_parts).strip(),
        )


def _record(provider: str, model: str, usage: Usage) -> None:
    METRICS.inc("creditlens_llm_tokens_total", value=usage.input_tokens,
                direction="input", provider=provider)
    METRICS.inc("creditlens_llm_tokens_total", value=usage.output_tokens,
                direction="output", provider=provider)
    METRICS.inc("creditlens_llm_cost_usd_total",
                value=usage.cost_usd(model, provider), provider=provider)
