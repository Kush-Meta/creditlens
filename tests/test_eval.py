"""The evaluation harness itself.

A silently-wrong eval metric is worse than no eval, so the metric functions
are tested against hand-computed values rather than only exercised end to end.
"""
from __future__ import annotations

import math

import pytest

from creditlens.eval import metrics
from creditlens.eval.dataset import EvalCase, RelevanceSpec, load_suite
from creditlens.eval.runner import relevant_chunk_ids, run_retrieval_ablation, run_suite


class TestMetrics:
    def test_recall_and_precision(self):
        assert metrics.recall_at_k([1, 2, 3, 4], {3, 9}) == pytest.approx(0.5)
        assert metrics.precision_at_k([1, 2, 3, 4], {3, 9}) == pytest.approx(0.25)

    def test_recall_normalization_accounts_for_the_ceiling(self):
        """With 100 relevant docs and k=8, raw recall cannot exceed 0.08."""
        retrieved = list(range(8))
        relevant = set(range(100))
        raw = metrics.recall_at_k(retrieved, relevant, 8)
        normalized = metrics.recall_ceiling_normalized(retrieved, relevant, 8)
        assert raw == pytest.approx(0.08)
        assert normalized == pytest.approx(1.0)

    def test_mrr(self):
        assert metrics.reciprocal_rank([5, 6, 3], {3}) == pytest.approx(1 / 3)
        assert metrics.reciprocal_rank([1, 2], {9}) == 0.0

    def test_ndcg_rewards_higher_ranks(self):
        early = metrics.ndcg_at_k([1, 9, 9, 9], {1}, 4)
        late = metrics.ndcg_at_k([9, 9, 9, 1], {1}, 4)
        assert early > late
        assert early == pytest.approx(1.0)

    def test_empty_relevant_set_is_nan_not_zero(self):
        assert math.isnan(metrics.recall_at_k([1], set()))
        assert math.isnan(metrics.ndcg_at_k([1], set()))

    def test_set_f1(self):
        scores = metrics.set_f1(["a", "b"], ["b", "c"])
        assert scores["precision"] == pytest.approx(0.5)
        assert scores["f1"] == pytest.approx(0.5)

    def test_compare_metric_values_parses_formatted_strings(self):
        assert metrics.compare_metric_values("3.42x", 3.4213)[0] is True
        assert metrics.compare_metric_values("18.6%", 20.0)[0] is False
        assert metrics.compare_metric_values("$1,234", 1234.0)[0] is True

    def test_compare_respects_written_precision(self):
        """0.2074 correctly rounded to "0.21x" must not be scored as an error."""
        assert metrics.compare_metric_values("0.21x", 0.2074, tolerance_pct=1.0)[0] is True

    def test_compare_rejects_unparseable(self):
        assert metrics.compare_metric_values("n/a", 1.0)[0] is False

    def test_aggregate_ignores_nans(self):
        summary = metrics.aggregate([1.0, float("nan"), 3.0])
        assert summary["n"] == 2
        assert summary["mean"] == pytest.approx(2.0)

    def test_accumulator_summary(self):
        accumulator = metrics.MetricAccumulator()
        for value in (0.5, 1.0):
            accumulator.add("score", value)
        accumulator.bump("cases")
        summary = accumulator.summary()
        assert summary["score"]["mean"] == pytest.approx(0.75)
        assert summary["counts"]["cases"] == 1


class TestDataset:
    def test_golden_suite_loads(self):
        cases = load_suite("golden")
        assert len(cases) >= 20
        assert len({c.id for c in cases}) == len(cases)

    def test_every_case_declares_something_checkable(self):
        for case in load_suite("golden"):
            assert (
                case.expected_tools or case.expected_metrics
                or case.relevance or case.must_mention
            ), f"{case.id} asserts nothing"

    def test_relevance_predicate(self):
        spec = RelevanceSpec(tickers=["NVCR"], items=["1A"], any_of=[["covenant"]])

        class Record:
            ticker, item_code, form_type = "NVCR", "1A", "10-K"
            text = "Our credit agreement contains a maximum net leverage covenant."

        assert spec.matches(Record()) is True
        Record.text = "Unrelated prose."
        assert spec.matches(Record()) is False

    def test_labels_resolve_against_the_corpus(self, session):
        """A case whose label set is empty measures nothing."""
        available = {"NVCR", "ARMT", "KSTR", "HRBG"}
        for case in load_suite("golden"):
            if not case.relevance or not set(case.tickers) <= available:
                continue
            assert relevant_chunk_ids(session, case.relevance), case.id


class TestSuiteRun:
    def test_suite_runs_and_reports_headline_metrics(self, session):
        from creditlens.agent.llm import OfflineEngine

        cases = [c for c in load_suite("golden") if c.requires_synthetic][:4]
        result = run_suite(
            session, engine=OfflineEngine(), cases=cases, persist=False
        )
        headline = result.metrics["headline"]
        assert headline["cases"] == len(cases)
        assert 0.0 <= headline["numeric_accuracy"] <= 1.0
        assert headline["citation_validity"] == 1.0
        assert headline["latency_p50_ms"] > 0

    def test_cases_for_missing_issuers_are_skipped_not_failed(self, session):
        from creditlens.agent.llm import OfflineEngine

        case = EvalCase(
            id="missing", question="What is NOSUCH's leverage?",
            category="test", tickers=["NOSUCH"],
        )
        result = run_suite(session, engine=OfflineEngine(), cases=[case], persist=False)
        assert result.n_cases == 0
        assert result.skipped

    def test_ablation_compares_configurations(self, session):
        cases = [c for c in load_suite("golden") if c.requires_synthetic][:4]
        ablation = run_retrieval_ablation(session, cases=cases)
        assert set(ablation["configurations"]) >= {"lexical_only", "dense_only"}
        for scores in ablation["configurations"].values():
            assert 0.0 <= (scores["ndcg_at_k"] or 0.0) <= 1.0

    def test_ablation_restores_configuration(self, session):
        from creditlens.config import get_settings

        before = get_settings().dense_weight
        cases = [c for c in load_suite("golden") if c.requires_synthetic][:2]
        run_retrieval_ablation(session, cases=cases)
        assert get_settings().dense_weight == before


class TestGeneratedSuite:
    """The generator must produce cases that actually measure something."""

    def test_every_generated_case_asserts_something(self):
        from creditlens.eval.generate import build_suite

        for case in build_suite():
            assert case.expected_tools or case.expected_metrics or case.relevance
            assert case.tickers
            assert case.generated is True

    def test_generation_is_deterministic(self):
        from creditlens.eval.generate import build_suite

        first = [c.id for c in build_suite()]
        second = [c.id for c in build_suite()]
        assert first == second

    def test_case_ids_are_unique(self):
        from creditlens.eval.generate import build_suite

        ids = [c.id for c in build_suite()]
        assert len(ids) == len(set(ids))

    def test_comparison_pairs_are_within_sector_and_prefer_contrast(self):
        """Pairs differ in profile wherever the sector offers a contrast.

        Utilities and REITs are uniformly structurally levered - that is a real
        property of those sectors, not a gap in the universe - so a same-profile
        pair there is the documented fallback rather than a defect.
        """
        from creditlens.eval.generate import build_comparison_cases
        from creditlens.ingest.universe import BY_TICKER, UNIVERSE

        profiles_by_sector: dict[str, set[str]] = {}
        for issuer in UNIVERSE:
            profiles_by_sector.setdefault(issuer.sector, set()).add(issuer.profile)

        for case in build_comparison_cases(UNIVERSE):
            left, right = (BY_TICKER[t] for t in case.tickers)
            assert left.sector == right.sector
            if len(profiles_by_sector[left.sector]) > 1:
                assert left.profile != right.profile

    def test_validate_drops_unscoreable_cases(self, session):
        from creditlens.eval.dataset import EvalCase, RelevanceSpec
        from creditlens.eval.generate import validate_suite

        cases = [
            EvalCase(id="absent-issuer", question="q", category="test", tickers=["NOSUCH"]),
            EvalCase(
                id="empty-labels", question="q", category="test", tickers=["NVCR"],
                relevance=RelevanceSpec(tickers=["NVCR"], any_of=[["zzzznotaword"]]),
            ),
            EvalCase(
                id="scoreable", question="q", category="test", tickers=["NVCR"],
                relevance=RelevanceSpec(tickers=["NVCR"], any_of=[["covenant"]]),
            ),
        ]
        kept, dropped = validate_suite(session, cases)
        assert [c.id for c in kept] == ["scoreable"]
        assert len(dropped) == 2

    def test_suites_combine_and_reject_duplicate_ids(self, tmp_path):
        import json

        from creditlens.eval.dataset import load_suite

        (tmp_path / "a.jsonl").write_text(
            json.dumps({"id": "x", "question": "q", "category": "c"}) + "\n")
        (tmp_path / "b.jsonl").write_text(
            json.dumps({"id": "y", "question": "q", "category": "c"}) + "\n")
        assert len(load_suite("a,b", directory=tmp_path)) == 2

        (tmp_path / "c.jsonl").write_text(
            json.dumps({"id": "x", "question": "q", "category": "c"}) + "\n")
        with pytest.raises(ValueError, match="duplicate case id"):
            load_suite("a,c", directory=tmp_path)


class TestUniverse:
    def test_universe_spans_the_credit_spectrum(self):
        from creditlens.ingest.universe import PROFILES, UNIVERSE, composition

        assert len(UNIVERSE) >= 40
        assert set(PROFILES) >= {"net_cash", "leveraged", "cyclical_stressed"}
        by_profile = composition()["by_profile"]
        # no single profile may dominate, or the corpus is analytically flat
        assert max(by_profile.values()) < len(UNIVERSE) * 0.5

    def test_tickers_are_unique(self):
        from creditlens.ingest.universe import UNIVERSE

        tickers = [i.ticker for i in UNIVERSE]
        assert len(tickers) == len(set(tickers))

    def test_filtering(self):
        from creditlens.ingest.universe import tickers

        assert "MSFT" in tickers(sector="Technology")
        assert "MSFT" not in tickers(sector="Utilities")
        assert all(t for t in tickers(profile="net_cash"))

    def test_fixture_source_needs_no_network(self):
        """CI gates on this suite, so it must build from the synthetic issuers."""
        from creditlens.eval.generate import SOURCES, build_suite, fixture_issuers

        assert set(SOURCES) == {"universe", "fixtures"}
        issuers = fixture_issuers()
        assert {i.ticker for i in issuers} == {"NVCR", "ARMT", "KSTR", "HRBG"}
        cases = build_suite(issuers)
        assert cases
        assert all(set(c.tickers) <= {"NVCR", "ARMT", "KSTR", "HRBG"} for c in cases)

    def test_shipped_suites_all_load(self):
        from creditlens.eval.dataset import load_suite

        combined = load_suite("golden,universe,fixtures")
        assert len(combined) > 200
        assert len({c.id for c in combined}) == len(combined)
