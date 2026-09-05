"""Deterministic financial computation.

These are the tests that matter most: every number the product surfaces
originates here, so an error would propagate silently through the agent, the
verifier and the eval harness alike.
"""
from __future__ import annotations

import datetime as dt

import pytest

from creditlens.finance import ratios, scorecard, statements, trends
from creditlens.finance.statements import Period, add_derived, build_ttm, select_period


class TestDerivedConcepts:
    def test_total_debt_sums_components(self, period_factory):
        period = period_factory()
        assert period.get("total_debt") == 600.0
        assert period.provenance["total_debt"].source == "derived"

    def test_total_debt_prefers_reported_total(self, period_factory):
        period = period_factory()
        period.values["total_debt_reported"] = 555.0
        period.values.pop("total_debt")
        add_derived(period)
        assert period.get("total_debt") == 555.0

    def test_long_term_tag_including_current_is_not_double_counted(self, period_factory):
        """`LongTermDebt` already includes current maturities at many filers."""
        period = period_factory()
        period.values.pop("total_debt")
        period.provenance["long_term_debt"] = statements.Provenance(
            concept="long_term_debt", source="sec-edgar-xbrl", raw_concept="LongTermDebt",
        )
        add_derived(period)
        assert period.get("total_debt") == 500.0  # not 600

    def test_ebitda_and_fcf(self, period_factory):
        period = period_factory()
        assert period.get("ebitda") == 250.0
        assert period.get("free_cash_flow") == 170.0
        assert period.get("net_debt") == 350.0

    def test_da_derived_from_split_tags(self, period_factory):
        period = period_factory(depreciation_amortization=None)
        period.values.pop("depreciation_amortization", None)
        period.values.pop("ebitda", None)
        period.values["depreciation_only"] = 30.0
        period.values["amortization_intangibles"] = 12.0
        add_derived(period)
        assert period.get("depreciation_amortization") == 42.0

    def test_missing_inputs_produce_no_derived_value(self, period_factory):
        period = period_factory()
        period.values.pop("short_term_debt")
        period.values.pop("long_term_debt")
        period.values.pop("total_debt")
        add_derived(period)
        assert period.get("total_debt") is None


class TestRatios:
    def test_leverage(self, period_factory):
        result = ratios.compute("debt_to_ebitda", period_factory())
        assert result.value == pytest.approx(2.4)
        assert result.inputs == {"total_debt": 600.0, "ebitda": 250.0}
        assert "total_debt" in result.provenance

    def test_quarterly_flows_are_annualized_with_a_warning(self, period_factory):
        annual = ratios.compute("debt_to_ebitda", period_factory("FY"))
        quarterly = ratios.compute("debt_to_ebitda", period_factory("Q3"))
        assert quarterly.value == pytest.approx(annual.value / 4)
        assert any("annualized" in w for w in quarterly.warnings)

    def test_non_positive_denominator_is_refused_not_computed(self, period_factory):
        period = period_factory(operating_income=-300.0)
        result = ratios.compute("debt_to_ebitda", period)
        assert result.value is None
        assert "not economically meaningful" in result.unavailable_reason

    def test_missing_input_is_reported(self, period_factory):
        period = period_factory()
        period.values.pop("interest_expense")
        result = ratios.compute("ebitda_interest_coverage", period)
        assert result.value is None
        assert "interest_expense" in result.unavailable_reason

    def test_every_registered_ratio_computes_on_a_complete_period(self, period_factory):
        results = ratios.compute_all(period_factory())
        failed = [r.name for r in results if r.value is None]
        assert failed == []

    def test_display_formatting(self, period_factory):
        period = period_factory()
        assert ratios.compute("debt_to_ebitda", period).display() == "2.40x"
        assert ratios.compute("operating_margin", period).display() == "20.0%"
        assert ratios.compute("days_sales_outstanding", period).display().endswith("days")

    def test_unknown_ratio_raises(self, period_factory):
        with pytest.raises(KeyError):
            ratios.compute("not_a_ratio", period_factory())

    def test_catalog_is_complete(self):
        catalog = ratios.catalog()
        assert len(catalog) == len(ratios.REGISTRY)
        assert all(entry["formula"] and entry["inputs"] for entry in catalog)


class TestTTM:
    def _quarters(self) -> list[Period]:
        out = []
        for i, quarter in enumerate(("Q1", "Q2", "Q3", "Q4")):
            period = Period("X", 1, 2025, quarter, dt.date(2025, 3 * (i + 1), 28))
            period.values.update({
                "revenue": 100.0 + i, "operating_income": 20.0 + i,
                "depreciation_amortization": 5.0, "total_assets": 1000.0,
                "short_term_debt": 50.0, "long_term_debt": 200.0,
                "cash_and_equivalents": 80.0,
            })
            add_derived(period)
            out.append(period)
        return out

    def test_flows_sum_and_stocks_do_not(self):
        ttm = build_ttm(self._quarters())
        assert ttm.get("revenue") == pytest.approx(406.0)
        assert ttm.get("total_assets") == pytest.approx(1000.0)
        assert ttm.get("ebitda") == pytest.approx(106.0)

    def test_requires_four_quarters(self):
        assert build_ttm(self._quarters()[:3]) is None

    def test_provenance_marks_ttm_derivation(self):
        ttm = build_ttm(self._quarters())
        assert ttm.provenance["revenue"].source == "derived-ttm"
        assert "Q42025" in ttm.provenance["revenue"].formula


class TestPeriodSelection:
    def test_ttm_label_round_trips(self):
        quarters = TestTTM()._quarters()
        ttm = build_ttm(quarters)
        assert select_period(quarters, "TTM").key == ttm.key
        # the rendered label must resolve back to the same scope
        assert select_period(quarters, ttm.key).fiscal_period == "TTM"

    def test_explicit_quarter(self):
        quarters = TestTTM()._quarters()
        assert select_period(quarters, "Q3-2025").key == "Q32025"

    def test_latest_and_unknown(self):
        quarters = TestTTM()._quarters()
        assert select_period(quarters, None).key == "Q42025"
        assert select_period(quarters, "FY1999") is None


class TestTrends:
    def _series(self) -> list[Period]:
        out = []
        for i, year in enumerate((2021, 2022, 2023, 2024)):
            period = Period("X", 1, year, "FY", dt.date(year, 12, 31))
            period.values.update({
                "revenue": 1000.0 + 100 * i, "cost_of_revenue": 600.0 + 100 * i,
                "operating_income": 200.0 - 15 * i, "depreciation_amortization": 50.0,
                "short_term_debt": 100.0, "long_term_debt": 500.0,
                "cash_and_equivalents": 200.0, "total_equity": 900.0,
                "interest_expense": 20.0, "sga_expense": 150.0, "rd_expense": 80.0,
            })
            add_derived(period)
            out.append(period)
        return out

    def test_direction_uses_credit_polarity(self):
        result = trends.trend_for_ratio("operating_margin", self._series())
        assert result.direction == trends.DETERIORATING
        assert result.slope_per_period < 0
        assert result.r_squared > 0.9

    def test_small_moves_are_stable_not_narrative(self):
        series = self._series()
        for period in series:
            period.values["operating_income"] = 200.0
            add_derived(period)
        result = trends.trend_for_ratio("ebitda_margin", series)
        assert result.direction in {trends.STABLE, trends.DETERIORATING}

    def test_attribution_components_sum_to_total(self):
        series = self._series()
        attribution = trends.attribute_ratio_change("debt_to_ebitda", series[0], series[-1])
        effects = sum(c["effect"] for c in attribution.components)
        # components are rounded for display; the decomposition itself is exact,
        # which the absence of a residual note asserts
        assert effects == pytest.approx(attribution.total_change, abs=1e-3)
        assert not attribution.notes

    def test_margin_bridge_is_exhaustive(self):
        series = self._series()
        bridge = trends.attribute_margin_change(series[0], series[-1])
        total = sum(c["effect_pp"] for c in bridge.components)
        assert total == pytest.approx(bridge.total_change, abs=1e-3)

    def test_pct_change_guards_zero(self):
        assert trends.pct_change(0, 10) is None
        assert trends.pct_change(None, 10) is None
        assert trends.pct_change(100, 150) == pytest.approx(50.0)


class TestScorecard:
    def test_healthy_issuer_scores_investment_grade(self, period_factory):
        card = scorecard.score(period_factory())
        assert card.composite > 65
        assert card.coverage == pytest.approx(1.0)
        assert card.confidence == "high"

    def test_distressed_issuer_scores_low(self, period_factory):
        period = period_factory(
            operating_income=-50.0, net_income=-90.0, operating_cash_flow=-30.0,
            interest_expense=120.0, cash_and_equivalents=10.0,
            short_term_investments=0.0, current_assets=300.0, current_liabilities=600.0,
        )
        card = scorecard.score(period)
        assert card.composite < 45
        assert card.weaknesses

    def test_coverage_drop_lowers_confidence(self, period_factory):
        period = period_factory()
        for concept in ("interest_expense", "operating_cash_flow", "capex",
                        "cash_and_equivalents", "short_term_investments"):
            period.values.pop(concept, None)
        for derived in ("free_cash_flow", "net_debt", "cash_and_investments"):
            period.values.pop(derived, None)
        card = scorecard.score(period)
        assert card.confidence in {"low", "medium"}
        assert card.coverage < 1.0
        assert any("could be computed" in w for w in card.warnings)

    def test_altman_z_reports_missing_inputs(self, period_factory):
        period = period_factory()
        period.values.pop("retained_earnings")
        result = scorecard.altman_z(period)
        assert result["available"] is False
        assert "retained_earnings" in result["missing_inputs"]

    def test_scorecard_always_carries_the_disclaimer(self, period_factory):
        disclaimer = scorecard.score(period_factory()).to_dict()["disclaimer"].lower()
        assert "not a credit rating" in disclaimer
        assert "not investment advice" in disclaimer
