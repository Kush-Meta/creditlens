"""OpenAI and every OpenAI-compatible endpoint.

One adapter covers a large share of the ecosystem, because so much of it
speaks the Chat Completions protocol: OpenAI itself, Azure OpenAI, Ollama,
vLLM, LM Studio, llama.cpp, Together, Fireworks, Groq, OpenRouter, DeepSeek and
Mistral's compatible endpoint. They differ only in `base_url`, credentials and
model name, so those are configuration rather than code.

Two protocol differences from Anthropic drive the translation:

* tool calls live on `message.tool_calls` rather than in the content, and the
  assistant message must be replayed with those calls attached;
* each result is its own message with `role="tool"` and a `tool_call_id`,
  rather than a batch of blocks inside one user message.

Local models are frequently weaker at tool use, so a model that answers in
prose instead of calling the terminal tool is handled by the orchestrator's
existing fallback rather than treated as a failure here.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from creditlens.agent.providers.base import (
    AssistantTurn,
    Conversation,
    LLMEngine,
    LLMError,
    LLMResponse,
    LLMUnavailable,
    ToolCall,
    ToolResultBatch,
    Usage,
    UserMessage,
    tool_schemas_to_openai,
)
from creditlens.observability import METRICS, get_logger

log = get_logger(__name__)

#: Reasoning models reject `temperature` and use `max_completion_tokens`.
REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class OpenAICompatibleEngine(LLMEngine):
    provider = "openai"

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        *,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        settings: Any | None = None,
    ):
        from creditlens.config import get_settings

        self.settings = settings or get_settings()
        self.provider = provider
        self.name = provider
        self.model = model or self.settings.model
        self.base_url = base_url
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailable(
                "the `openai` package is not installed; `pip install openai`"
            ) from exc
        try:
            self.client = OpenAI(
                api_key=api_key or self._resolve_key(),
                base_url=base_url,
                timeout=self.settings.llm_timeout_s,
                max_retries=self.settings.llm_max_retries,
            )
        except Exception as exc:
            raise LLMUnavailable(f"could not construct the {provider} client: {exc}") from exc

    def _resolve_key(self) -> str | None:
        import os

        # Local servers accept any non-empty key; hosted ones need a real one.
        for variable in (f"{self.provider.upper()}_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(variable)
            if value:
                return value
        return "not-needed" if self.base_url else None

    @property
    def _is_reasoning_model(self) -> bool:
        return any(self.model.startswith(prefix) for prefix in REASONING_PREFIXES)

    # -- wire format ------------------------------------------------------
    def _to_messages(self, system: str, conversation: Conversation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in conversation:
            if isinstance(turn, UserMessage):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.response.text or None,
                }
                if turn.response.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in turn.response.tool_calls
                    ]
                messages.append(message)
            elif isinstance(turn, ToolResultBatch):
                for result in turn.results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content,
                    })
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_messages(system, conversation),
        }
        limit = max_tokens or self.settings.max_tokens
        if self._is_reasoning_model:
            kwargs["max_completion_tokens"] = limit
        else:
            kwargs["max_tokens"] = limit
        if tools:
            kwargs["tools"] = tool_schemas_to_openai(tools)
            kwargs["tool_choice"] = (
                {"type": "function", "function": {"name": force_tool}}
                if force_tool else "auto"
            )

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if any(hint in lowered for hint in
                   ("api key", "unauthorized", "authentication", "connection", "not found")):
                raise LLMUnavailable(f"{self.provider}: {message}") from exc
            raise LLMError(f"{self.provider}: {message}") from exc

        return self._normalize(response)

    def _normalize(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            raw_arguments = getattr(call.function, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                # Weaker models emit malformed argument JSON. Surface it as a
                # tool error the loop can recover from, not a crash.
                log.warning("unparseable tool arguments",
                            extra={"tool": call.function.name, "raw": raw_arguments[:200]})
                arguments = {"__malformed_arguments__": raw_arguments[:500]}
            calls.append(ToolCall(
                id=call.id or f"call_{index}", name=call.function.name, arguments=arguments
            ))

        raw_usage = getattr(response, "usage", None)
        details = getattr(raw_usage, "prompt_tokens_details", None)
        completion_details = getattr(raw_usage, "completion_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0 if raw_usage else 0
        usage = Usage(
            # cached tokens are reported inside prompt_tokens, so subtract to
            # avoid billing the same tokens twice at two different rates
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0 if raw_usage else 0,
            cache_read_tokens=cached,
            reasoning_tokens=(
                getattr(completion_details, "reasoning_tokens", 0) or 0
                if completion_details else 0
            ),
            requests=1,
        )
        _record(self.provider, self.model, usage)

        finish = getattr(choice, "finish_reason", "stop") or "stop"
        return LLMResponse(
            text=(getattr(message, "content", None) or "").strip(),
            tool_calls=calls,
            stop_reason={"tool_calls": "tool_use", "stop": "end_turn"}.get(finish, finish),
            usage=usage,
            raw=None,  # the neutral turn is sufficient to replay this protocol
        )


def _record(provider: str, model: str, usage: Usage) -> None:
    METRICS.inc("creditlens_llm_tokens_total", value=usage.input_tokens,
                direction="input", provider=provider)
    METRICS.inc("creditlens_llm_tokens_total", value=usage.output_tokens,
                direction="output", provider=provider)
    METRICS.inc("creditlens_llm_cost_usd_total",
                value=usage.cost_usd(model, provider), provider=provider)
