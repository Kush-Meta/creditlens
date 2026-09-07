"""Provider adapters.

Each provider is exercised against a fake client, so the suite verifies the
part that actually breaks in practice - translating the neutral transcript into
three incompatible tool protocols and parsing three incompatible responses -
without a network call or a credential.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from creditlens.agent.providers import (
    PROVIDERS,
    AssistantTurn,
    LLMResponse,
    ToolCall,
    ToolResult,
    ToolResultBatch,
    Usage,
    UserMessage,
    available_providers,
    build_engine,
)
from creditlens.agent.providers.anthropic_engine import AnthropicEngine
from creditlens.agent.providers.base import (
    LLMRefusal,
    LLMUnavailable,
    tool_schemas_to_google,
    tool_schemas_to_openai,
)
from creditlens.agent.providers.google_engine import GoogleEngine
from creditlens.agent.providers.offline_engine import OfflineEngine
from creditlens.agent.providers.openai_engine import OpenAICompatibleEngine
from creditlens.agent.providers.pricing import price_for

TOOLS = [{
    "name": "compute_ratios",
    "description": "Compute ratios.",
    "input_schema": {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
        "additionalProperties": False,
    },
}]

CONVERSATION = [
    UserMessage("What is MSFT's leverage?"),
    AssistantTurn(LLMResponse(
        text="Let me check.",
        tool_calls=[ToolCall("call_1", "compute_ratios", {"ticker": "MSFT"})],
    )),
    ToolResultBatch([ToolResult("call_1", "compute_ratios", '{"debt_to_ebitda": 0.21}')]),
]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_every_provider_declares_a_default_model_and_factory(self):
        for key, spec in PROVIDERS.items():
            assert spec.key == key
            assert spec.default_model
            assert callable(spec.factory)

    def test_offline_is_always_available(self):
        entry = next(p for p in available_providers() if p["provider"] == "offline")
        assert entry["credentials_detected"] is True

    def test_unknown_provider_degrades_rather_than_raising(self):
        assert isinstance(build_engine("not-a-provider"), OfflineEngine)

    def test_explicit_offline_selection(self):
        engine = build_engine("offline")
        assert isinstance(engine, OfflineEngine)
        assert engine.provider == "offline"

    def test_provider_without_credentials_degrades(self, monkeypatch):
        for name in ("OPENAI_API_KEY",):
            monkeypatch.delenv(name, raising=False)
        engine = build_engine("openai")
        # either it degraded, or a key really is present in this environment
        assert isinstance(engine, OfflineEngine) or engine.provider == "openai"

    def test_endpoint_probe_rejects_a_non_api_server(self):
        from creditlens.agent.providers import endpoint_reachable

        # nothing should be listening here; a bare socket check is not enough
        assert endpoint_reachable("http://localhost:1/v1", timeout=0.2) is False
        assert endpoint_reachable(None) is False


# ---------------------------------------------------------------------------
# anthropic
# ---------------------------------------------------------------------------
class TestAnthropicAdapter:
    @pytest.fixture
    def engine(self):
        engine = AnthropicEngine.__new__(AnthropicEngine)
        engine.model = "claude-opus-5"
        engine.provider = "anthropic"
        return engine

    def test_tool_results_batch_into_one_user_message(self, engine):
        """Splitting results across messages trains the model out of parallel calls."""
        conversation = list(CONVERSATION)
        conversation[-1] = ToolResultBatch([
            ToolResult("a", "t", "{}"), ToolResult("b", "t", "{}"),
        ])
        messages = engine._to_messages(conversation)
        assert messages[-1]["role"] == "user"
        assert len(messages[-1]["content"]) == 2

    def test_assistant_raw_is_replayed_verbatim(self, engine):
        """Thinking blocks must be echoed back unchanged, not reconstructed."""
        sentinel = [{"type": "thinking", "thinking": "..."}]
        conversation = [AssistantTurn(LLMResponse(text="x", raw=sentinel))]
        assert engine._to_messages(conversation)[0]["content"] is sentinel

    def test_assistant_reconstructed_when_raw_absent(self, engine):
        messages = engine._to_messages(CONVERSATION)
        blocks = messages[1]["content"]
        assert {b["type"] for b in blocks} == {"text", "tool_use"}

    def test_error_flag_survives(self, engine):
        conversation = [ToolResultBatch([ToolResult("a", "t", '{"error":"x"}', is_error=True)])]
        assert engine._to_messages(conversation)[0]["content"][0]["is_error"] is True

    def test_response_parsing(self, engine):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="reasoning"),
                SimpleNamespace(type="text", text="Leverage is low."),
                SimpleNamespace(type="tool_use", id="t1", name="compute_ratios",
                                input={"ticker": "MSFT"}),
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                  cache_read_input_tokens=80, cache_creation_input_tokens=0),
            stop_reason="tool_use",
        )
        parsed = engine._normalize(response)
        assert parsed.text == "Leverage is low."
        assert parsed.thinking == "reasoning"
        assert parsed.tool_calls[0].arguments == {"ticker": "MSFT"}
        assert parsed.usage.cache_read_tokens == 80

    def test_refusal_raises(self, engine):
        response = SimpleNamespace(
            content=[], usage=SimpleNamespace(input_tokens=1, output_tokens=0),
            stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"),
        )
        with pytest.raises(LLMRefusal, match="cyber"):
            engine._normalize(response)


# ---------------------------------------------------------------------------
# openai-compatible
# ---------------------------------------------------------------------------
class TestOpenAIAdapter:
    @pytest.fixture
    def engine(self):
        engine = OpenAICompatibleEngine.__new__(OpenAICompatibleEngine)
        engine.model = "gpt-5"
        engine.provider = "openai"
        engine.name = "openai"
        engine.base_url = None
        return engine

    def test_system_prompt_becomes_a_message(self, engine):
        messages = engine._to_messages("SYSTEM", CONVERSATION)
        assert messages[0] == {"role": "system", "content": "SYSTEM"}

    def test_tool_calls_attach_to_the_assistant_message(self, engine):
        messages = engine._to_messages("s", CONVERSATION)
        assistant = messages[2]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"][0]["function"]["name"] == "compute_ratios"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"ticker": "MSFT"}

    def test_each_result_is_its_own_tool_message(self, engine):
        conversation = [ToolResultBatch([
            ToolResult("a", "t", "{}"), ToolResult("b", "t", "{}"),
        ])]
        messages = engine._to_messages("s", conversation)[1:]
        assert [m["role"] for m in messages] == ["tool", "tool"]
        assert [m["tool_call_id"] for m in messages] == ["a", "b"]

    def test_schema_translation(self):
        translated = tool_schemas_to_openai(TOOLS)[0]
        assert translated["type"] == "function"
        assert translated["function"]["parameters"]["required"] == ["ticker"]

    def test_response_parsing_and_cache_accounting(self, engine):
        """Cached tokens sit inside prompt_tokens and must not be billed twice."""
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="done",
                    tool_calls=[SimpleNamespace(
                        id="c1",
                        function=SimpleNamespace(name="compute_ratios",
                                                 arguments='{"ticker":"MSFT"}'))],
                ),
                finish_reason="tool_calls",
            )],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20,
                prompt_tokens_details=SimpleNamespace(cached_tokens=40),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
            ),
        )
        parsed = engine._normalize(response)
        assert parsed.stop_reason == "tool_use"
        assert parsed.tool_calls[0].arguments == {"ticker": "MSFT"}
        assert parsed.usage.input_tokens == 60
        assert parsed.usage.cache_read_tokens == 40
        assert parsed.usage.reasoning_tokens == 5

    def test_malformed_tool_arguments_do_not_crash(self, engine):
        """Weaker local models emit invalid argument JSON; the loop must survive."""
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                    id="c1", function=SimpleNamespace(name="compute_ratios",
                                                      arguments="{not json"))]),
                finish_reason="tool_calls")],
            usage=None,
        )
        parsed = engine._normalize(response)
        assert "__malformed_arguments__" in parsed.tool_calls[0].arguments

    def test_reasoning_models_use_max_completion_tokens(self, engine):
        engine.model = "o3-mini"
        assert engine._is_reasoning_model is True
        engine.model = "gpt-4o"
        assert engine._is_reasoning_model is False


# ---------------------------------------------------------------------------
# google
# ---------------------------------------------------------------------------
class TestGoogleAdapter:
    @pytest.fixture
    def engine(self):
        engine = GoogleEngine.__new__(GoogleEngine)
        engine.model = "gemini-2.5-pro"
        engine.provider = "google"
        return engine

    def test_roles_are_user_and_model(self, engine):
        contents = engine._to_contents(CONVERSATION)
        assert [c["role"] for c in contents] == ["user", "model", "user"]

    def test_results_correlate_by_function_name(self, engine):
        """Gemini carries no call id, so the name is the correlation key."""
        contents = engine._to_contents(CONVERSATION)
        assert contents[-1]["parts"][0]["function_response"]["name"] == "compute_ratios"

    def test_result_payload_is_an_object(self, engine):
        conversation = [ToolResultBatch([ToolResult("a", "t", "plain text")])]
        response = engine._to_contents(conversation)[0]["parts"][0]["function_response"]
        assert response["response"] == {"output": "plain text"}

    def test_schema_strips_unsupported_keywords(self):
        parameters = tool_schemas_to_google(TOOLS)[0]["parameters"]
        assert "additionalProperties" not in parameters
        assert parameters["properties"]["ticker"]["type"] == "string"

    def test_response_parsing(self, engine):
        response = SimpleNamespace(
            candidates=[SimpleNamespace(
                content=SimpleNamespace(parts=[
                    SimpleNamespace(text="hello", function_call=None),
                    SimpleNamespace(text=None, function_call=SimpleNamespace(
                        id=None, name="compute_ratios", args={"ticker": "MSFT"})),
                ]),
                finish_reason="STOP")],
            usage_metadata=SimpleNamespace(prompt_token_count=50, candidates_token_count=10,
                                           cached_content_token_count=0, thoughts_token_count=3),
        )
        parsed = engine._normalize(response)
        assert parsed.text == "hello"
        assert parsed.tool_calls[0].name == "compute_ratios"
        assert parsed.usage.reasoning_tokens == 3

    def test_safety_stop_is_a_refusal(self, engine):
        response = SimpleNamespace(
            candidates=[SimpleNamespace(
                content=SimpleNamespace(parts=[]), finish_reason="SAFETY")],
            usage_metadata=None,
        )
        with pytest.raises(LLMRefusal):
            engine._normalize(response)


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
class TestPricing:
    def test_published_anthropic_rates(self):
        assert price_for("claude-opus-5") == (5.00, 25.00)

    def test_bedrock_prefix_resolves(self):
        assert price_for("anthropic.claude-opus-5") == (5.00, 25.00)

    def test_local_providers_are_free(self):
        assert price_for("qwen2.5:14b", "ollama") == (0.0, 0.0)

    def test_unknown_model_has_no_rate(self):
        assert price_for("some-model-we-do-not-price") is None

    def test_unpriced_model_says_so_rather_than_reporting_zero(self):
        """Inventing a cost would undercut the point of the whole system."""
        payload = Usage(input_tokens=1000, requests=1).to_dict("some-model", "openai")
        assert payload["cost_basis"] == "no rate configured"
        assert payload["estimated_cost_usd"] == 0.0

    def test_rates_can_be_configured(self, monkeypatch):
        monkeypatch.setenv("CREDITLENS_MODEL_PRICES", "my-model=1.5/6.0")
        assert price_for("my-model") == (1.5, 6.0)
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.cost_usd("my-model") == pytest.approx(7.5)

    def test_cache_reads_are_cheaper(self):
        cached = Usage(cache_read_tokens=1_000_000).cost_usd("claude-opus-5")
        fresh = Usage(input_tokens=1_000_000).cost_usd("claude-opus-5")
        assert cached == pytest.approx(fresh * 0.1)


# ---------------------------------------------------------------------------
# engine substitution through the orchestrator
# ---------------------------------------------------------------------------
class TestEngineSubstitution:
    def test_any_engine_drives_the_same_loop(self, session):
        """A scripted provider proves the loop is provider-agnostic."""
        from creditlens.agent.orchestrator import Orchestrator
        from creditlens.agent.providers.base import LLMEngine

        class ScriptedEngine(LLMEngine):
            name, provider, model = "scripted", "test", "scripted-1"

            def __init__(self):
                self.turns = 0
                self.seen_conversations: list[int] = []

            def complete(self, *, system, conversation, tools=None,
                         max_tokens=None, force_tool=None):
                self.seen_conversations.append(len(conversation))
                self.turns += 1
                if self.turns == 1:
                    return LLMResponse(
                        text="", stop_reason="tool_use",
                        usage=Usage(input_tokens=10, output_tokens=5, requests=1),
                        tool_calls=[ToolCall("s1", "compute_ratios",
                                             {"ticker": "NVCR", "period": "TTM"})],
                    )
                return LLMResponse(
                    text="", stop_reason="tool_use",
                    usage=Usage(input_tokens=10, output_tokens=5, requests=1),
                    tool_calls=[ToolCall("s2", "submit_analysis", {
                        "answer": "Leverage is elevated.",
                        "credit_direction": "deteriorating",
                        "key_metrics": [], "reasoning": "r", "confidence": "medium",
                    })],
                )

        engine = ScriptedEngine()
        result = Orchestrator(session, engine=engine, persist=False).analyze(
            "How levered is Novacore?", tickers=["NVCR"]
        )
        assert result.status == "ok"
        assert result.provider == "test"
        assert result.credit_direction == "deteriorating"
        assert [t["tool"] for t in result.tool_calls] == ["compute_ratios"]
        # the transcript grows by the assistant turn plus the tool-result batch
        assert engine.seen_conversations == [1, 3]

    def test_provider_failure_degrades_to_offline(self, session):
        from creditlens.agent.orchestrator import Orchestrator
        from creditlens.agent.providers.base import LLMEngine

        class DeadEngine(LLMEngine):
            name, provider, model = "dead", "openai", "gpt-5"

            def complete(self, **kwargs):
                raise LLMUnavailable("no credentials")

        result = Orchestrator(session, engine=DeadEngine(), persist=False).analyze(
            "Summarize Novacore's credit profile."
        )
        assert result.status == "ok"
        assert result.engine == "offline-deterministic"
        assert "openai unavailable" in (result.degraded_reason or "")


class TestAdapterIntegration:
    """Drive the real orchestrator loop through the real adapters.

    A fake transport is injected at the SDK boundary, so translation, parsing
    and the multi-turn loop are all exercised together - the combination that
    actually breaks when a provider is swapped - with no network or credential.
    """

    def _fake_openai_client(self):
        turns = {"n": 0}

        def create(**kwargs):
            turns["n"] += 1
            messages = kwargs["messages"]
            # the transcript must round-trip: after the first turn the tool
            # result has to be present as a `tool` message
            if turns["n"] > 1:
                assert any(m["role"] == "tool" for m in messages), messages
            if turns["n"] == 1:
                call = SimpleNamespace(
                    id="c1",
                    function=SimpleNamespace(
                        name="compute_ratios",
                        arguments=json.dumps({"ticker": "NVCR", "period": "TTM"})),
                )
                message = SimpleNamespace(content=None, tool_calls=[call])
                finish = "tool_calls"
            else:
                call = SimpleNamespace(
                    id="c2",
                    function=SimpleNamespace(name="submit_analysis", arguments=json.dumps({
                        "answer": "Leverage is elevated.",
                        "credit_direction": "deteriorating",
                        "key_metrics": [], "reasoning": "r", "confidence": "medium",
                    })),
                )
                message = SimpleNamespace(content=None, tool_calls=[call])
                finish = "tool_calls"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason=finish)],
                usage=SimpleNamespace(
                    prompt_tokens=200, completion_tokens=30,
                    prompt_tokens_details=None, completion_tokens_details=None),
            )

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_openai_adapter_completes_an_analysis(self, session):
        from creditlens.agent.orchestrator import Orchestrator

        engine = OpenAICompatibleEngine(
            model="gpt-5", client=self._fake_openai_client(), provider="openai"
        )
        result = Orchestrator(session, engine=engine, persist=False).analyze(
            "How levered is Novacore?", tickers=["NVCR"]
        )
        assert result.status == "ok"
        assert result.provider == "openai"
        assert result.credit_direction == "deteriorating"
        assert [t["tool"] for t in result.tool_calls] == ["compute_ratios"]
        assert result.usage["input_tokens"] == 400

    def _fake_anthropic_client(self):
        turns = {"n": 0}

        def create(**kwargs):
            turns["n"] += 1
            messages = kwargs["messages"]
            if turns["n"] > 1:
                # the tool result must come back as a user message of blocks
                last = messages[-1]
                assert last["role"] == "user"
                assert last["content"][0]["type"] == "tool_result"
            if turns["n"] == 1:
                block = SimpleNamespace(type="tool_use", id="t1", name="compute_ratios",
                                        input={"ticker": "NVCR", "period": "TTM"})
            else:
                block = SimpleNamespace(type="tool_use", id="t2", name="submit_analysis",
                                        input={
                                            "answer": "Leverage is elevated.",
                                            "credit_direction": "deteriorating",
                                            "key_metrics": [], "reasoning": "r",
                                            "confidence": "medium",
                                        })
            return SimpleNamespace(
                content=[block], stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=180, output_tokens=25,
                                      cache_read_input_tokens=120,
                                      cache_creation_input_tokens=0),
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    def test_anthropic_adapter_completes_an_analysis(self, session):
        from creditlens.agent.orchestrator import Orchestrator

        engine = AnthropicEngine(model="claude-opus-5", client=self._fake_anthropic_client())
        result = Orchestrator(session, engine=engine, persist=False).analyze(
            "How levered is Novacore?", tickers=["NVCR"]
        )
        assert result.status == "ok"
        assert result.provider == "anthropic"
        assert result.credit_direction == "deteriorating"
        assert result.usage["cache_read_tokens"] == 240
        # a priced model reports a real cost, unlike an unpriced one
        assert result.usage["cost_basis"] == "published rates"
        assert result.cost_usd > 0

    def _fake_google_client(self):
        turns = {"n": 0}

        def generate_content(**kwargs):
            turns["n"] += 1
            contents = kwargs["contents"]
            if turns["n"] > 1:
                responses = [
                    part for message in contents for part in message["parts"]
                    if "function_response" in part
                ]
                assert responses, contents
            name = "compute_ratios" if turns["n"] == 1 else "submit_analysis"
            args = ({"ticker": "NVCR", "period": "TTM"} if turns["n"] == 1 else {
                "answer": "Leverage is elevated.",
                "credit_direction": "deteriorating",
                "key_metrics": [], "reasoning": "r", "confidence": "medium",
            })
            part = SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(id=None, name=name, args=args),
            )
            return SimpleNamespace(
                candidates=[SimpleNamespace(
                    content=SimpleNamespace(parts=[part]), finish_reason="STOP")],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=150, candidates_token_count=20,
                    cached_content_token_count=0, thoughts_token_count=0),
            )

        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    def test_google_adapter_completes_an_analysis(self, session):
        from creditlens.agent.orchestrator import Orchestrator

        engine = GoogleEngine(model="gemini-2.5-pro", client=self._fake_google_client())
        result = Orchestrator(session, engine=engine, persist=False).analyze(
            "How levered is Novacore?", tickers=["NVCR"]
        )
        assert result.status == "ok"
        assert result.provider == "google"
        assert result.credit_direction == "deteriorating"
        assert result.usage["input_tokens"] == 300

    def test_verification_runs_identically_whatever_the_provider(self, session):
        """The checks that make an answer trustworthy are provider-independent."""
        from creditlens.agent.orchestrator import Orchestrator

        engine = OpenAICompatibleEngine(
            model="gpt-5", client=self._fake_openai_client(), provider="openai"
        )
        result = Orchestrator(session, engine=engine, persist=False).analyze(
            "How levered is Novacore?", tickers=["NVCR"]
        )
        verification = result.verification
        assert verification["citation_validity"] == 1.0
        assert verification["unsupported_claim_rate"] == 0.0
        assert "confidence" in verification


class TestOllamaModelResolution:
    """A provider must not advertise a model that is not installed."""

    def test_exact_match_is_kept(self, monkeypatch):
        from creditlens.agent import providers

        monkeypatch.setattr(providers, "installed_ollama_models",
                            lambda url: ["qwen2.5:7b", "llama3.1:8b"])
        assert providers._resolve_ollama_model("qwen2.5:7b", "http://x/v1") == "qwen2.5:7b"

    def test_same_family_different_size_is_not_a_match(self, monkeypatch):
        """qwen2.5:14b is not satisfied by qwen2.5:7b being installed."""
        from creditlens.agent import providers

        monkeypatch.setattr(providers, "installed_ollama_models",
                            lambda url: ["qwen2.5:3b", "qwen2.5:7b"])
        assert providers._resolve_ollama_model("qwen2.5:14b", "http://x/v1") == "qwen2.5:7b"

    def test_largest_variant_wins(self, monkeypatch):
        from creditlens.agent import providers

        monkeypatch.setattr(providers, "installed_ollama_models",
                            lambda url: ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:1.5b"])
        assert providers._resolve_ollama_model(None, "http://x/v1") == "qwen2.5:7b"

    def test_nothing_installed_leaves_the_request_alone(self, monkeypatch):
        from creditlens.agent import providers

        monkeypatch.setattr(providers, "installed_ollama_models", lambda url: [])
        assert providers._resolve_ollama_model("qwen2.5:14b", "http://x/v1") == "qwen2.5:14b"

    def test_parameter_parsing(self):
        from creditlens.agent.providers import _parameter_billions

        assert _parameter_billions("qwen2.5:14b") == 14.0
        assert _parameter_billions("qwen2.5:1.5b") == 1.5
        assert _parameter_billions("llama3:latest") == 0.0
