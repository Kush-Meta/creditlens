"""Google Gemini provider.

Gemini's protocol differs from both others in ways that matter here:

* the system prompt is a separate `system_instruction`, not a message;
* tool calls and their results are `parts` inside content, and results go back
  under the `"user"` role rather than a dedicated tool role;
* function call ids are not carried, so results are correlated **by function
  name**, which the adapter reproduces faithfully;
* the parameter schema dialect rejects several JSON Schema keywords, which
  `tool_schemas_to_google` strips.
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
    tool_schemas_to_google,
)
from creditlens.observability import METRICS, get_logger

log = get_logger(__name__)


class GoogleEngine(LLMEngine):
    name = "google"
    provider = "google"

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        settings: Any | None = None,
    ):
        from creditlens.config import get_settings

        self.settings = settings or get_settings()
        self.model = model or self.settings.model
        if client is not None:
            self.client = client
            return
        try:
            from google import genai
        except ImportError as exc:
            raise LLMUnavailable(
                "the `google-genai` package is not installed; `pip install google-genai`"
            ) from exc
        try:
            import os

            self.client = genai.Client(
                api_key=api_key or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            )
        except Exception as exc:
            raise LLMUnavailable(f"could not construct the Gemini client: {exc}") from exc

    # -- wire format ------------------------------------------------------
    def _to_contents(self, conversation: Conversation) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for turn in conversation:
            if isinstance(turn, UserMessage):
                contents.append({"role": "user", "parts": [{"text": turn.text}]})
            elif isinstance(turn, AssistantTurn):
                parts: list[dict[str, Any]] = []
                if turn.response.text:
                    parts.append({"text": turn.response.text})
                for call in turn.response.tool_calls:
                    parts.append({
                        "function_call": {"name": call.name, "args": call.arguments}
                    })
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            elif isinstance(turn, ToolResultBatch):
                contents.append({"role": "user", "parts": [
                    {
                        "function_response": {
                            "name": result.name,
                            "response": _as_object(result.content, result.is_error),
                        }
                    }
                    for result in turn.results
                ]})
        return contents

    def complete(
        self,
        *,
        system: str,
        conversation: Conversation,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        force_tool: str | None = None,
    ) -> LLMResponse:
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens or self.settings.max_tokens,
        }
        if tools:
            config["tools"] = [{"function_declarations": tool_schemas_to_google(tools)}]
            config["tool_config"] = {
                "function_calling_config": (
                    {"mode": "ANY", "allowed_function_names": [force_tool]}
                    if force_tool else {"mode": "AUTO"}
                )
            }

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self._to_contents(conversation),
                config=config,
            )
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if any(hint in lowered for hint in
                   ("api key", "permission", "unauthenticated", "not found", "connection")):
                raise LLMUnavailable(f"google: {message}") from exc
            raise LLMError(f"google: {message}") from exc

        return self._normalize(response)

    def _normalize(self, response: Any) -> LLMResponse:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        candidates = getattr(response, "candidates", None) or []
        for index, part in enumerate(_parts(candidates)):
            function_call = getattr(part, "function_call", None)
            if function_call is not None and getattr(function_call, "name", None):
                calls.append(ToolCall(
                    # Gemini carries no call id, so one is synthesised; results
                    # are correlated by function name on the way back.
                    id=getattr(function_call, "id", None) or f"gemini_{index}",
                    name=function_call.name,
                    arguments=dict(getattr(function_call, "args", None) or {}),
                ))
            elif getattr(part, "text", None):
                text_parts.append(part.text)

        metadata = getattr(response, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(metadata, "prompt_token_count", 0) or 0 if metadata else 0,
            output_tokens=getattr(metadata, "candidates_token_count", 0) or 0 if metadata else 0,
            cache_read_tokens=(
                getattr(metadata, "cached_content_token_count", 0) or 0 if metadata else 0
            ),
            reasoning_tokens=getattr(metadata, "thoughts_token_count", 0) or 0 if metadata else 0,
            requests=1,
        )
        _record(self.provider, self.model, usage)

        finish = str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
        if finish.upper().endswith("SAFETY"):
            from creditlens.agent.providers.base import LLMRefusal

            raise LLMRefusal("Gemini declined the request (safety)")
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            usage=usage,
            raw=None,
        )


def _parts(candidates: Any) -> list[Any]:
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    return list(getattr(content, "parts", None) or [])


def _as_object(content: str, is_error: bool) -> dict[str, Any]:
    """Gemini requires a function response to be an object, not a string."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"output": content}
    if not isinstance(parsed, dict):
        parsed = {"output": parsed}
    if is_error:
        parsed = {**parsed, "is_error": True}
    return parsed


def _record(provider: str, model: str, usage: Usage) -> None:
    METRICS.inc("creditlens_llm_tokens_total", value=usage.input_tokens,
                direction="input", provider=provider)
    METRICS.inc("creditlens_llm_tokens_total", value=usage.output_tokens,
                direction="output", provider=provider)
    METRICS.inc("creditlens_llm_cost_usd_total",
                value=usage.cost_usd(model, provider), provider=provider)
