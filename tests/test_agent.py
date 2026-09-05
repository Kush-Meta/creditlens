"""Agent layer: tools, the evidence ledger, verification and the loop."""
from __future__ import annotations

import pytest

from creditlens.agent import tools as tool_mod
from creditlens.agent.llm import OfflineEngine, Usage
from creditlens.agent.orchestrator import Orchestrator
from creditlens.agent.tools import ToolError
from creditlens.agent.verifier import extract_numeric_claims, verify


class TestTools:
    def test_every_schema_has_an_implementation(self):
        assert {s["name"] for s in tool_mod.TOOL_SCHEMAS} == set(tool_mod.TOOL_IMPLS)

    def test_schemas_are_strict_objects(self):
        for schema in tool_mod.TOOL_SCHEMAS:
            body = schema["input_schema"]
            assert body["type"] == "object"
            assert body.get("additionalProperties") is False
            assert schema["description"]

    def test_list_companies(self, tool_ctx):
        payload = tool_mod.tool_list_companies(tool_ctx)
        assert payload["count"] >= 4
        assert all(c["is_synthetic"] for c in payload["companies"])

    def test_get_financials_includes_a_ttm_row(self, tool_ctx):
        payload = tool_mod.tool_get_financials(tool_ctx, ticker="NVCR", periods=4)
        assert any(row["period"].startswith("TTM") for row in payload["periods"])
        assert payload["revenue_growth"]

    def test_compute_ratios_registers_evidence(self, tool_ctx):
        payload = tool_mod.tool_compute_ratios(tool_ctx, ticker="NVCR", period="TTM")
        assert payload["ratios"]
        assert any("debt_to_ebitda" in key for key in tool_ctx.ledger.numbers)

    def test_compute_ratios_rejects_unknown_names(self, tool_ctx):
        with pytest.raises(ToolError, match="unknown ratio"):
            tool_mod.tool_compute_ratios(tool_ctx, ticker="NVCR", ratios=["made_up"])

    def test_unknown_ticker_lists_the_alternatives(self, tool_ctx):
        with pytest.raises(ToolError, match="Available issuers"):
            tool_mod.tool_get_financials(tool_ctx, ticker="NOPE")

    def test_compare_periods_returns_an_exact_attribution(self, tool_ctx):
        payload = tool_mod.tool_compare_periods(
            tool_ctx, ticker="NVCR", metric="operating_margin"
        )
        assert payload["change"]["direction"] in {"improving", "deteriorating", "stable", "mixed"}
        assert "margin_bridge" in payload

    def test_metric_trend_classifies_direction(self, tool_ctx):
        payload = tool_mod.tool_metric_trend(
            tool_ctx, ticker="NVCR", metric="net_debt_to_ebitda", lookback=6
        )
        assert payload["direction"] == "deteriorating"
        assert len(payload["series"]) == 6

    def test_metric_trend_accepts_raw_concepts(self, tool_ctx):
        payload = tool_mod.tool_metric_trend(tool_ctx, ticker="ARMT", metric="revenue")
        assert payload["direction"] in {"increasing", "decreasing"}

    def test_credit_scorecard_shape(self, tool_ctx):
        payload = tool_mod.tool_credit_scorecard(tool_ctx, ticker="ARMT", period="TTM")
        assert payload["composite_score_0_100"] > 60
        assert payload["disclaimer"]

    def test_compare_companies_needs_two(self, tool_ctx):
        with pytest.raises(ToolError, match="at least two"):
            tool_mod.tool_compare_companies(tool_ctx, tickers=["NVCR"])

    def test_compare_companies_picks_a_winner_per_ratio(self, tool_ctx):
        payload = tool_mod.tool_compare_companies(
            tool_ctx, tickers=["ARMT", "HRBG"], period="TTM"
        )
        winners = payload["stronger_on_each_ratio"]
        assert winners["current_ratio"] == "ARMT"

    def test_search_filings_assigns_citation_labels(self, tool_ctx):
        payload = tool_mod.tool_search_filings(
            tool_ctx, query="covenant headroom", tickers=["NVCR"], top_k=4
        )
        labels = [p["citation"] for p in payload["passages"]]
        assert labels == [f"C{i}" for i in range(1, len(labels) + 1)]

    def test_data_coverage_explains_what_is_missing(self, tool_ctx):
        payload = tool_mod.tool_data_coverage(tool_ctx, ticker="NVCR")
        assert payload["ratios_available"] > 0
        assert "concepts_available" in payload

    def test_execute_returns_errors_instead_of_raising(self, tool_ctx):
        result, is_error = tool_mod.execute(tool_ctx, "compute_ratios", {"ticker": "NOPE"})
        assert is_error is True
        assert "error" in result

    def test_execute_rejects_unknown_tools(self, tool_ctx):
        result, is_error = tool_mod.execute(tool_ctx, "no_such_tool", {})
        assert is_error is True
        assert "available_tools" in result

    def test_execute_handles_bad_arguments(self, tool_ctx):
        _result, is_error = tool_mod.execute(tool_ctx, "compute_ratios", {"bogus": 1})
        assert is_error is True


class TestEvidenceLedger:
    def test_passages_get_stable_labels(self, session, ledger):
        from creditlens.retrieval import hybrid

        hits = hybrid.search(session, "liquidity", tickers=["NVCR"], top_k=3).chunks
        first = ledger.add_passage(hits[0])
        again = ledger.add_passage(hits[0])
        assert first.label == again.label == "C1"
        assert ledger.add_passage(hits[1]).label == "C2"

    def test_synthetic_flag_propagates(self, session, ledger):
        from creditlens.retrieval import hybrid

        ledger.add_passage(hybrid.search(session, "liquidity", tickers=["NVCR"]).chunks[0])
        assert ledger.has_synthetic() is True

    def test_citations_filtered_by_label(self, session, ledger):
        from creditlens.retrieval import hybrid

        for hit in hybrid.search(session, "liquidity", tickers=["NVCR"], top_k=3).chunks:
            ledger.add_passage(hit)
        assert [c["label"] for c in ledger.citations({"C2"})] == ["C2"]


class TestVerifier:
    """The adversarial suite: the verifier must catch invented figures."""

    @pytest.fixture
    def grounded(self, tool_ctx):
        tool_mod.tool_compute_ratios(tool_ctx, ticker="NVCR", period="TTM")
        tool_mod.tool_search_filings(
            tool_ctx, query="senior notes mature refinance covenant leverage",
            tickers=["NVCR"], top_k=5,
        )
        return tool_ctx.ledger

    def _status(self, text: str, ledger) -> str:
        claims = verify(narrative=text, ledger=ledger).numeric_claims
        assert claims, f"no numeric claim extracted from {text!r}"
        return claims[0].status

    def test_correct_figure_is_verified(self, grounded):
        value = grounded.numbers["NVCR:TTM2026:debt_to_ebitda"].value
        assert self._status(f"Gross leverage stands at {value:.2f}x.", grounded) == "verified"

    def test_invented_figure_is_unsupported(self, grounded):
        assert self._status("Net leverage is 4.10x.", grounded) == "unsupported"

    def test_figure_attached_to_the_wrong_metric_is_not_verified(self, grounded):
        """1.29x may be a real debt/equity, but it is not the current ratio."""
        equity = grounded.numbers["NVCR:TTM2026:debt_to_equity"].value
        assert self._status(f"Debt to equity is {equity:.2f}x.", grounded) == "verified"
        assert self._status(f"The current ratio was {equity:.2f}x.", grounded) != "verified"

    def test_correct_rounding_is_not_a_contradiction(self, grounded):
        """"18.6%" asserts 18.6 +/- 0.05, not 18.600000."""
        value = grounded.numbers["NVCR:TTM2026:operating_margin"].value
        assert self._status(f"Operating margin was {value:.1f}%.", grounded) == "verified"

    def test_invented_dollar_amount_is_unsupported(self, grounded):
        assert self._status("Revenue reached $47.3 billion.", grounded) == "unsupported"

    def test_invalid_citation_is_flagged(self, grounded):
        report = verify(narrative="See [C99] for detail.", ledger=grounded)
        assert report.invalid_citations == ["C99"]
        assert report.citation_validity == 0.0

    def test_confidence_cannot_exceed_the_evidence(self, grounded):
        report = verify(
            narrative="Net leverage is 4.10x and coverage 9.8x.",
            ledger=grounded, stated_confidence="high",
        )
        assert report.confidence == "low"

    def test_extraction_handles_every_unit(self):
        claims = extract_numeric_claims(
            "Leverage 3.42x, revenue $4.2 billion, margin 18.6%, "
            "down 180 bps, DSO 55 days."
        )
        units = {c.unit for c in claims}
        assert {"multiple", "USD", "percent", "bps", "days"} <= units

    def test_years_are_not_treated_as_money(self):
        claims = extract_numeric_claims("In FY2024 the company refinanced.")
        assert all(c.unit != "USD" for c in claims)


class TestOrchestrator:
    def test_offline_run_produces_a_structured_answer(self, session):
        result = Orchestrator(session, engine=OfflineEngine(), persist=False).analyze(
            "Has Novacore's credit quality deteriorated over the last four quarters?"
        )
        assert result.status == "ok"
        assert result.credit_direction == "deteriorating"
        assert result.key_metrics and result.citations
        assert result.engine == "offline-deterministic"

    def test_every_number_is_traceable(self, session):
        result = Orchestrator(session, engine=OfflineEngine(), persist=False).analyze(
            "Summarize Harbridge's credit profile."
        )
        assert result.verification["unsupported_claim_rate"] < 0.1
        assert result.verification["citation_validity"] == 1.0

    def test_unknown_issuer_is_refused_not_hallucinated(self, session):
        result = Orchestrator(session, engine=OfflineEngine(), persist=False).analyze(
            "What is Tesla's leverage?"
        )
        assert "tesla" in result.answer.lower()
        assert result.key_metrics == []
        assert result.credit_direction == "not_applicable"

    def test_synthetic_data_is_always_disclosed(self, session):
        result = Orchestrator(session, engine=OfflineEngine(), persist=False).analyze(
            "Summarize Aramont's credit profile."
        )
        assert result.contains_synthetic_data is True

    def test_tool_budget_is_enforced(self, session, monkeypatch):
        from creditlens.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "max_tool_calls", 2)
        result = Orchestrator(session, engine=OfflineEngine(), persist=False).analyze(
            "Summarize Novacore's credit profile."
        )
        assert len(result.tool_calls) <= 2
        assert result.status == "ok"

    def test_iteration_budget_terminates_the_loop(self, session, monkeypatch):
        from creditlens.config import get_settings

        monkeypatch.setattr(get_settings(), "max_tool_iterations", 1)
        result = Orchestrator(session, engine=OfflineEngine(), persist=False).analyze(
            "Summarize Novacore's credit profile."
        )
        assert result.status in {"ok", "incomplete"}

    def test_llm_failure_degrades_instead_of_erroring(self, session):
        from creditlens.agent.llm import LLMEngine, LLMError

        class Broken(LLMEngine):
            name, model = "broken", "broken"

            def complete(self, **kwargs):
                raise LLMError("simulated outage")

        result = Orchestrator(session, engine=Broken(), persist=False).analyze(
            "Summarize Novacore's credit profile."
        )
        assert result.status == "ok"
        assert result.engine == "offline-deterministic"
        assert "unavailable" in (result.degraded_reason or "")

    def test_trace_records_the_tool_spans(self, session):
        result = Orchestrator(session, engine=OfflineEngine(), persist=False).analyze(
            "Compare Aramont and Kestra on leverage."
        )
        names = {span["name"] for span in result.trace}
        assert "analysis" in names
        assert any(name.startswith("tool.") for name in names)


class TestUsageAccounting:
    def test_cost_uses_the_published_rates(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.cost_usd("claude-opus-5") == pytest.approx(30.0)

    def test_cached_reads_are_cheaper(self):
        cached = Usage(cache_read_tokens=1_000_000).cost_usd("claude-opus-5")
        fresh = Usage(input_tokens=1_000_000).cost_usd("claude-opus-5")
        assert cached == pytest.approx(fresh * 0.1)

    def test_unknown_model_costs_nothing_rather_than_guessing(self):
        assert Usage(input_tokens=1000).cost_usd("some-other-model") == 0.0
