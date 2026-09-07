"""Provider registry.

Adding a provider is a class and a table entry. The orchestrator, the tools,
the verifier and the evaluation harness are all unaware of which one is in use.

Selection order for `build_engine()`:

1. an explicitly requested provider;
2. the configured `llm_provider`;
3. auto-detection, if configured provider is `auto` - the first provider whose
   credentials actually resolve;
4. the deterministic offline engine, which always works.

Construction failures never propagate: an unreachable or unconfigured provider
degrades to the offline engine with the reason recorded on the analysis, so the
service answers a narrower question rather than returning an error.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
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
    ToolResult,
    ToolResultBatch,
    Usage,
    UserMessage,
)
from creditlens.agent.providers.offline_engine import OfflineEngine, classify_intents
from creditlens.observability import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    #: environment variables that indicate the provider is configured; empty
    #: means it needs no credentials
    credential_env: tuple[str, ...]
    default_model: str
    factory: Callable[..., LLMEngine]
    #: OpenAI-compatible endpoints that only differ by base URL
    base_url: str | None = None
    notes: str = ""


def _anthropic(**kwargs: Any) -> LLMEngine:
    from creditlens.agent.providers.anthropic_engine import AnthropicEngine

    return AnthropicEngine(**kwargs)


def _anthropic_backend(backend: str) -> Callable[..., LLMEngine]:
    def factory(**kwargs: Any) -> LLMEngine:
        from creditlens.agent.providers.anthropic_engine import AnthropicEngine

        return AnthropicEngine(backend=backend, **kwargs)

    return factory


def _openai_like(provider: str, base_url: str | None) -> Callable[..., LLMEngine]:
    def factory(**kwargs: Any) -> LLMEngine:
        from creditlens.agent.providers.openai_engine import OpenAICompatibleEngine

        kwargs.setdefault("base_url", base_url)
        if provider == "ollama":
            kwargs["model"] = _resolve_ollama_model(kwargs.get("model"), base_url)
        return OpenAICompatibleEngine(provider=provider, **kwargs)

    return factory


#: Preferred local models, in order, among those actually installed. Tool use is
#: the constraint: a model that cannot call tools cannot drive the agent loop.
_OLLAMA_PREFERENCE = ("qwen2.5", "qwen3", "llama3.1", "llama3.2", "mistral", "gemma3")


def installed_ollama_models(base_url: str | None) -> list[str]:
    if not base_url:
        return []
    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/').removesuffix('/v1')}/api/tags", timeout=1.5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return []


def _resolve_ollama_model(requested: str | None, base_url: str | None) -> str | None:
    """Name a model that is actually pulled.

    Advertising a default that is not installed makes /health report a provider
    as ready when the first request will fail. Better to report what is there.
    """
    installed = installed_ollama_models(base_url)
    if not installed:
        return requested
    # Exact match only: "qwen2.5:14b" is not satisfied by "qwen2.5:7b" being
    # installed, and treating it as satisfied is how a provider reports itself
    # ready for a model that will 404 on the first request.
    if requested and requested in installed:
        return requested
    for prefix in _OLLAMA_PREFERENCE:
        # Within a family prefer the largest installed variant: 7b beats 3b, and
        # tool-following on small local models degrades quickly with size.
        matches = sorted(
            (n for n in installed if n.startswith(prefix)),
            key=_parameter_billions, reverse=True,
        )
        if matches:
            chosen = matches[0]
            if requested and requested != chosen:
                log.info("requested ollama model is not installed; using an installed one",
                         extra={"requested": requested, "using": chosen})
            return chosen
    return installed[0]


def _parameter_billions(model: str) -> float:
    """Parameter count parsed from an Ollama tag, for ranking variants."""
    match = re.search(r"(\d+(?:\.\d+)?)b\b", model.lower())
    return float(match.group(1)) if match else 0.0


def _google(**kwargs: Any) -> LLMEngine:
    from creditlens.agent.providers.google_engine import GoogleEngine

    return GoogleEngine(**kwargs)


def _offline(**kwargs: Any) -> LLMEngine:
    kwargs.pop("model", None)
    return OfflineEngine()


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic",
        ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "claude-opus-5", _anthropic,
        notes="also resolves `ant auth login` profiles, so an unset key is not proof of no credentials",
    ),
    "bedrock": ProviderSpec(
        "bedrock", "Anthropic on Amazon Bedrock",
        ("AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
        "anthropic.claude-opus-5", _anthropic_backend("bedrock"),
    ),
    "vertex": ProviderSpec(
        "vertex", "Anthropic on Google Vertex AI",
        ("GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT"),
        "claude-opus-5", _anthropic_backend("vertex"),
    ),
    "openai": ProviderSpec(
        "openai", "OpenAI",
        ("OPENAI_API_KEY",), "gpt-5", _openai_like("openai", None),
    ),
    "azure": ProviderSpec(
        "azure", "Azure OpenAI",
        ("AZURE_OPENAI_API_KEY",), "gpt-5",
        _openai_like("azure", os.environ.get("AZURE_OPENAI_ENDPOINT")),
        notes="set AZURE_OPENAI_ENDPOINT",
    ),
    "google": ProviderSpec(
        "google", "Google Gemini",
        ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "gemini-2.5-pro", _google,
    ),
    "groq": ProviderSpec(
        "groq", "Groq", ("GROQ_API_KEY",), "llama-3.3-70b-versatile",
        _openai_like("groq", "https://api.groq.com/openai/v1"),
    ),
    "together": ProviderSpec(
        "together", "Together AI", ("TOGETHER_API_KEY",),
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        _openai_like("together", "https://api.together.xyz/v1"),
    ),
    "openrouter": ProviderSpec(
        "openrouter", "OpenRouter", ("OPENROUTER_API_KEY",), "openai/gpt-5",
        _openai_like("openrouter", "https://openrouter.ai/api/v1"),
    ),
    "deepseek": ProviderSpec(
        "deepseek", "DeepSeek", ("DEEPSEEK_API_KEY",), "deepseek-chat",
        _openai_like("deepseek", "https://api.deepseek.com/v1"),
    ),
    "mistral": ProviderSpec(
        "mistral", "Mistral", ("MISTRAL_API_KEY",), "mistral-large-latest",
        _openai_like("mistral", "https://api.mistral.ai/v1"),
    ),
    "ollama": ProviderSpec(
        "ollama", "Ollama (local)", (), "qwen2.5:14b",
        _openai_like("ollama", "http://localhost:11434/v1"),
        base_url="http://localhost:11434/v1",
        notes="runs locally; no key, no token cost",
    ),
    "vllm": ProviderSpec(
        "vllm", "vLLM / self-hosted", (), "local-model",
        _openai_like("vllm", os.environ.get("CREDITLENS_VLLM_BASE_URL", "http://localhost:8000/v1")),
        base_url=os.environ.get("CREDITLENS_VLLM_BASE_URL", "http://localhost:8000/v1"),
    ),
    "offline": ProviderSpec(
        "offline", "Deterministic engine (no model)", (), "offline-deterministic-v1", _offline,
        notes="always available; used as the fallback for every other provider",
    ),
}


def endpoint_reachable(base_url: str | None, timeout: float = 1.0) -> bool:
    """Is an OpenAI-compatible server actually answering at this endpoint?

    Keyless local providers (Ollama, vLLM) have no credential to detect, so
    "no key required" would otherwise read as "ready" even when nothing is
    listening. A bare TCP connect is not enough either: developer machines have
    something on port 8000 more often than not, and an unrelated web app would
    register as a working model server. So the probe asks for the model list
    and requires a JSON answer.
    """
    if not base_url:
        return False
    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    except Exception:
        return False
    if response.status_code >= 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and "data" in payload


def _is_ready(spec: ProviderSpec) -> bool:
    if spec.key == "offline":
        return True
    if spec.credential_env:
        return any(os.environ.get(name) for name in spec.credential_env)
    return endpoint_reachable(spec.base_url)


def available_providers() -> list[dict[str, Any]]:
    """Every provider with whether it is actually usable from here."""
    out: list[dict[str, Any]] = []
    for spec in PROVIDERS.values():
        configured = _is_ready(spec)
        out.append({
            "provider": spec.key,
            "label": spec.label,
            "default_model": spec.default_model,
            "credentials_detected": configured,
            "detection": "credentials" if spec.credential_env else "endpoint reachability",
            "credential_env": list(spec.credential_env),
            "notes": spec.notes,
        })
    return out


#: Order tried when `llm_provider` is "auto". Hosted frontier models first,
#: then local, then the deterministic engine.
AUTO_ORDER = ("anthropic", "openai", "google", "bedrock", "vertex",
              "groq", "together", "openrouter", "deepseek", "mistral", "ollama")


def build_engine(
    provider: str | None = None, model: str | None = None, **kwargs: Any
) -> LLMEngine:
    """Construct the configured engine, degrading to offline when unavailable."""
    from creditlens.config import get_settings

    settings = get_settings()
    requested = (provider or settings.llm_provider or "auto").lower()

    if requested == "auto":
        return _auto_select(settings, model, **kwargs)
    if requested not in PROVIDERS:
        log.warning("unknown provider; using the deterministic engine",
                    extra={"provider": requested, "known": sorted(PROVIDERS)})
        return OfflineEngine()

    spec = PROVIDERS[requested]
    try:
        return spec.factory(model=model or settings.model or spec.default_model, **kwargs)
    except Exception as exc:
        log.warning("provider unavailable; using the deterministic engine",
                    extra={"provider": requested, "error": str(exc)})
        return OfflineEngine()


def _auto_select(settings: Any, model: str | None, **kwargs: Any) -> LLMEngine:
    for key in AUTO_ORDER:
        spec = PROVIDERS[key]
        if not _is_ready(spec):
            continue
        try:
            engine = spec.factory(model=model or spec.default_model, **kwargs)
        except Exception as exc:
            log.info("auto-select skipped a provider",
                     extra={"provider": key, "error": str(exc)})
            continue
        log.info("auto-selected provider", extra={"provider": key, "model": engine.model})
        return engine
    log.info("no provider credentials resolved; using the deterministic engine")
    return OfflineEngine()


def probe_engine(provider: str | None = None) -> dict[str, Any]:
    """Cheap liveness probe for /health, without spending output tokens."""
    from creditlens.config import get_settings

    settings = get_settings()
    requested = (provider or settings.llm_provider or "auto").lower()
    detected = [p for p in available_providers() if p["credentials_detected"]]

    if requested == "offline":
        return {"provider": "offline", "model": "offline-deterministic-v1",
                "reachable": True, "providers_detected": [p["provider"] for p in detected]}

    engine = build_engine(provider)
    if isinstance(engine, OfflineEngine):
        return {
            "provider": requested, "model": settings.model, "reachable": False,
            "fallback": "offline-deterministic",
            "providers_detected": [p["provider"] for p in detected],
            "hint": "no credentials resolved for the configured provider",
        }
    return {
        "provider": engine.provider, "model": engine.model, "reachable": True,
        "engine": engine.name,
        "providers_detected": [p["provider"] for p in detected],
    }


__all__ = [
    "PROVIDERS",
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
    "probe_engine",
]
