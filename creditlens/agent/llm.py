"""Compatibility surface for the provider layer.

The engines live in `creditlens.agent.providers`; this module re-exports them
so callers have one obvious import site and older imports keep working.
"""
from __future__ import annotations

from creditlens.agent.providers import (
    PROVIDERS,
    AssistantTurn,
    Conversation,
    LLMEngine,
    LLMError,
    LLMRefusal,
    LLMResponse,
    LLMUnavailable,
    OfflineEngine,
    ProviderSpec,
    ToolCall,
    ToolResult,
    ToolResultBatch,
    Usage,
    UserMessage,
    available_providers,
    build_engine,
    classify_intents,
    probe_engine,
)
from creditlens.agent.providers.anthropic_engine import AnthropicEngine
from creditlens.agent.providers.pricing import PUBLISHED as PRICING
from creditlens.agent.providers.pricing import price_for

__all__ = [
    "PRICING",
    "PROVIDERS",
    "AnthropicEngine",
    "AssistantTurn",
    "Conversation",
    "LLMEngine",
    "LLMError",
    "LLMRefusal",
    "LLMResponse",
    "LLMUnavailable",
    "OfflineEngine",
    "ProviderSpec",
    "ToolCall",
    "ToolResult",
    "ToolResultBatch",
    "Usage",
    "UserMessage",
    "available_providers",
    "build_engine",
    "classify_intents",
    "price_for",
    "probe_engine",
]
