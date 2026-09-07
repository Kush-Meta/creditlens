"""Data lineage.

This surface exists to be right about where numbers come from, so the tests
check the property that matters: every leaf of a lineage tree terminates in
either a filed fact with an accession, or an explicit statement that the value
was derived and by what formula. Nothing may appear from nowhere.
"""
from __future__ import annotations

import pytest

from creditlens.agent.provenance import (
    LineageResolver,
    _find_concept,
    _find_metric,
    _find_period,
    _find_ticker,
    answer,
    origin_label,
)


class TestEntityResolution:
    KNOWN = ["AAPL", "ORCL", "MSFT", "F", "T", "M", "NVCR"]
    NAMES = {"F": "FORD MOTOR CO", "T": "AT&T INC.", "ORCL": "ORACLE CORP",
             "NVCR": "Novacore Industries, Inc."}

    def test_ticker_match(self):
        assert _find_ticker("What about ORCL?", self.KNOWN, self.NAMES) == "ORCL"

    def test_company_name_match(self):
        assert _find_ticker("Ford's leverage", self.KNOWN, self.NAMES) == "F"

    def test_short_tickers_do_not_match_inside_contractions(self):
        """`\\bT\\b` finds AT&T in "can't", which is how this first broke."""
        assert _find_ticker("Why can't you compute Ford's leverage?",
                            self.KNOWN, self.NAMES) == "F"

    def test_short_ticker_matches_when_standalone(self):
        assert _find_ticker("What data do you have for F?", self.KNOWN, self.NAMES) == "F"

    def test_short_ticker_not_matched_in_lowercase_prose(self):
        assert _find_ticker("the market fell", self.KNOWN, self.NAMES) is None

    def test_no_entity(self):
        assert _find_ticker("how does verification work", self.KNOWN, self.NAMES) is None

    @pytest.mark.parametrize("question,expected", [
        ("Where does debt-to-EBITDA come from?", "debt_to_ebitda"),
        ("what is net leverage", "net_debt_to_ebitda"),
        ("show me interest coverage", "ebitda_interest_coverage"),
        ("the current ratio please", "current_ratio"),
        ("operating margin trend", "operating_margin"),
    ])
    def test_metric_phrases_including_hyphenated(self, question, expected):
        assert _find_metric(question) == expected

    def test_concepts(self):
        assert _find_concept("where does revenue come from") == "revenue"
        assert _find_concept("total debt please") == "total_debt"

    @pytest.mark.parametrize("question,expected", [
        ("leverage for FY2024", "FY2024"),
        ("revenue in Q3 2025", "Q32025"),
        ("TTM figures", "TTM"),
        ("no period here", None),
    ])
    def test_periods(self, question, expected):
        assert _find_period(question) == expected

    def test_origin_labels_are_plain_language(self):
        assert "reported" in origin_label("sec-edgar-xbrl")
        assert "fourth" in origin_label("derived-q4") or "minus" in origin_label("derived-q4")
        assert "cumulative" in origin_label("derived-from-ytd (6M less Q1)")
        assert "carried forward" in origin_label("carried-forward-from-2024-12-31")


class TestLineage:
    def test_ratio_resolves_to_its_inputs(self, session):
        root = LineageResolver(session, "NVCR").resolve_ratio("debt_to_ebitda", "TTM")
        assert root.kind == "computed"
        assert root.formula
        assert {c.label for c in root.children} == {"total_debt", "ebitda"}

    def test_every_leaf_is_accounted_for(self, session):
        """No value may appear without either a source filing or a formula."""
        root = LineageResolver(session, "NVCR").resolve_ratio("net_debt_to_ebitda", "TTM")
        for leaf in root.leaves():
            assert leaf.kind in {"reported", "derived", "missing"}
            assert leaf.accession or leaf.formula or leaf.origin, leaf.label

    def test_ttm_expands_into_its_quarters(self, session):
        root = LineageResolver(session, "NVCR").resolve_ratio("debt_to_ebitda", "TTM")
        ebitda = next(c for c in root.children if c.label == "ebitda")
        assert ebitda.kind == "derived"
        assert len(ebitda.children) == 4
        assert all(child.period.startswith("Q") for child in ebitda.children)

    def test_derived_concepts_expand_into_components(self, session):
        root = LineageResolver(session, "NVCR").resolve_ratio("debt_to_ebitda", "TTM")
        ebitda = next(c for c in root.children if c.label == "ebitda")
        quarter = ebitda.children[0]
        assert {c.label for c in quarter.children} >= {"operating_income"}

    def test_synthetic_values_are_flagged_through_the_tree(self, session):
        root = LineageResolver(session, "NVCR").resolve_ratio("debt_to_ebitda", "TTM")
        assert any(node.is_synthetic for node in root.leaves())

    def test_unavailable_metric_is_explained_not_invented(self, session):
        resolver = LineageResolver(session, "NVCR")
        root = resolver.resolve_ratio("days_inventory", "TTM")
        if root.value is None:
            assert root.kind == "missing"
            assert root.note

    def test_unknown_ticker_yields_a_missing_root(self, session):
        root = LineageResolver(session, "NOSUCH").resolve_ratio("debt_to_ebitda")
        assert root.kind == "missing"

    def test_recursion_is_bounded(self, session):
        root = LineageResolver(session, "NVCR").resolve_ratio("net_debt_to_ebitda", "TTM")

        def depth(node, level=0):
            return max([depth(c, level + 1) for c in node.children], default=level)

        assert depth(root) <= 8

    def test_serialisable(self, session):
        import json

        root = LineageResolver(session, "NVCR").resolve_ratio("current_ratio", "TTM")
        assert json.loads(json.dumps(root.to_dict()))["label"]


class TestAnswers:
    def test_metric_question(self, session):
        result = answer(session, "Where does Novacore's debt-to-EBITDA come from?")
        assert result.intent == "metric_lineage"
        assert result.lineage is not None
        assert "ratio engine" in " ".join(result.detail)

    def test_unavailable_metric_explains_why(self, session):
        result = answer(session, "Why can't you compute Novacore's days inventory?")
        assert result.intent in {"metric_lineage", "coverage"}
        assert result.summary

    def test_coverage_question(self, session):
        result = answer(session, "What data do you have for NVCR?")
        assert result.intent == "coverage"
        assert "periods" in result.summary

    def test_corpus_question(self, session):
        result = answer(session, "What companies do you have?")
        assert result.intent == "corpus"
        assert result.table
        assert result.contains_synthetic is True

    def test_filings_question(self, session):
        result = answer(session, "What filings do you have for NVCR?")
        assert result.intent == "filings"
        assert result.table
        assert {"form", "period", "accession"} <= set(result.table[0])

    @pytest.mark.parametrize("question,topic", [
        ("How do you verify the numbers?", "verification"),
        ("How does retrieval work?", "retrieval"),
        ("How are Q4 figures derived?", "derived"),
        ("Where does the data come from?", "source"),
    ])
    def test_pipeline_questions(self, session, question, topic):
        result = answer(session, question)
        assert result.intent == f"pipeline:{topic}"
        assert result.detail

    def test_every_answer_offers_followups_or_content(self, session):
        for question in ["What companies do you have?",
                         "Where does Novacore's revenue come from?",
                         "How do you verify the numbers?"]:
            result = answer(session, question)
            assert result.summary
            assert result.followups or result.table or result.lineage

    def test_answers_are_serialisable(self, session):
        import json

        payload = answer(session, "Where does Novacore's leverage come from?").to_dict()
        assert json.loads(json.dumps(payload))["intent"]

    def test_engine_is_labelled_deterministic(self, session):
        """The reply must never be attributable to a model."""
        assert answer(session, "What companies do you have?").engine == "deterministic-lineage"


class TestProvenanceApi:
    def test_endpoint(self, client):
        payload = client.post(
            "/api/provenance",
            json={"question": "Where does Novacore's debt-to-EBITDA come from?"},
        ).json()
        assert payload["intent"] == "metric_lineage"
        assert payload["lineage"]["children"]

    def test_metric_lineage_endpoint(self, client):
        payload = client.get("/api/provenance/metric/NVCR/debt_to_ebitda?period=TTM").json()
        assert payload["label"]
        assert payload["children"]

    def test_unknown_ratio_is_404(self, client):
        assert client.get("/api/provenance/metric/NVCR/nope").status_code == 404

    def test_unknown_ticker_is_404(self, client):
        assert client.get("/api/provenance/metric/ZZZZ/debt_to_ebitda").status_code == 404

    def test_question_is_validated(self, client):
        assert client.post("/api/provenance", json={"question": "x"}).status_code == 422

    def test_assets_are_content_hashed(self, client):
        """A cached script running against new HTML fails in a confusing way."""
        html = client.get("/").text
        assert "/app.js?v=" in html
        assert client.get("/app.js").headers["cache-control"] == "no-cache"
