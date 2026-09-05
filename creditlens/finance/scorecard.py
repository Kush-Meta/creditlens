"""A transparent, deterministic credit scorecard.

This is an *internal heuristic model*, not a credit rating and not investment
advice. It exists so the narrative layer has a stable, reproducible anchor:
the same inputs always produce the same score, the weights are visible, and
every factor reports the ratio, the anchor curve, and its own contribution.

Design choices worth defending in review:

* **Piecewise-linear anchors** rather than a fitted model. With no labelled
  default data in the repo, a fitted model would be false precision; anchors
  encode published rating-agency style thresholds and are auditable line by line.
* **Weight renormalization over available factors**, with the share of weight
  actually covered reported as `coverage`. A score built from 40% of the
  factors must not look as confident as one built from 100%.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from creditlens.finance import ratios as ratio_mod
from creditlens.finance.statements import Period

DISCLAIMER = (
    "Internal heuristic scorecard for analytical comparison only. Not a credit "
    "rating, not a rating-agency output, and not investment advice."
)


@dataclass(frozen=True)
class Factor:
    ratio: str
    weight: float
    #: (metric value, score) anchor points, ascending by metric value
    anchors: tuple[tuple[float, float], ...]
    rationale: str = ""


def _interp(value: float, anchors: Sequence[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation, clamped at the ends."""
    if value <= anchors[0][0]:
        return anchors[0][1]
    if value >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return anchors[-1][1]


FACTORS: tuple[Factor, ...] = (
    Factor("debt_to_ebitda", 0.18, (
        (0.0, 100), (1.0, 92), (2.0, 80), (3.0, 66), (4.0, 52),
        (5.0, 40), (6.0, 28), (8.0, 12), (12.0, 0),
    ), "Gross leverage vs. cash earnings - the primary rating driver."),
    Factor("debt_to_capital", 0.12, (
        (0.0, 100), (20.0, 88), (35.0, 74), (50.0, 58), (65.0, 40),
        (80.0, 22), (95.0, 5),
    ), "Structural reliance on debt in the capital stack."),
    Factor("ebitda_interest_coverage", 0.15, (
        (0.0, 0), (1.0, 15), (2.0, 32), (3.0, 48), (5.0, 65),
        (8.0, 80), (15.0, 92), (30.0, 100),
    ), "Headroom to service interest from operating earnings."),
    Factor("fcf_interest_coverage", 0.10, (
        (-5.0, 0), (0.0, 20), (1.0, 40), (2.0, 56), (4.0, 72),
        (8.0, 88), (15.0, 100),
    ), "Interest cover after capital spending - harder to manage than EBITDA."),
    Factor("cfo_to_debt", 0.09, (
        (-10.0, 0), (0.0, 12), (10.0, 35), (20.0, 55), (35.0, 74),
        (60.0, 90), (100.0, 100),
    ), "Cash generation against the debt stack; a rating-agency staple."),
    Factor("fcf_margin", 0.06, (
        (-20.0, 0), (-5.0, 20), (0.0, 40), (5.0, 58), (12.0, 76),
        (20.0, 90), (35.0, 100),
    ), "Self-funding capacity."),
    Factor("current_ratio", 0.07, (
        (0.5, 5), (0.8, 25), (1.0, 45), (1.3, 62), (1.8, 80),
        (2.5, 92), (4.0, 100),
    ), "Near-term obligations vs. near-term assets."),
    Factor("cash_to_debt", 0.08, (
        (0.0, 5), (10.0, 28), (25.0, 50), (50.0, 70), (100.0, 88),
        (200.0, 100),
    ), "Liquidity buffer against the debt stack."),
    Factor("ebitda_margin", 0.08, (
        (-10.0, 0), (0.0, 18), (8.0, 38), (15.0, 55), (25.0, 74),
        (40.0, 90), (60.0, 100),
    ), "Structural earnings power and shock absorption."),
    Factor("return_on_assets", 0.07, (
        (-15.0, 0), (0.0, 22), (3.0, 42), (6.0, 60), (10.0, 78),
        (18.0, 92), (30.0, 100),
    ), "Efficiency of the asset base backing the debt."),
)

_RATING_BANDS: tuple[tuple[float, str, str], ...] = (
    (88.0, "aa or higher", "Very strong: minimal leverage, deep coverage and liquidity"),
    (78.0, "a", "Strong: comfortable coverage with conservative leverage"),
    (66.0, "bbb", "Adequate investment-grade profile with moderate leverage"),
    (56.0, "bb", "Speculative: leverage or coverage constrains flexibility"),
    (44.0, "b", "Highly speculative: thin coverage, meaningful refinancing risk"),
    (0.0, "ccc or lower", "Weak: cash generation may not sustain the capital structure"),
)


@dataclass
class FactorScore:
    ratio: str
    label: str
    category: str
    weight: float
    effective_weight: float
    value: float | None
    display: str
    score: float | None
    contribution: float | None
    rationale: str
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "label": self.label,
            "category": self.category,
            "value": None if self.value is None else round(self.value, 4),
            "display": self.display,
            "score_0_100": None if self.score is None else round(self.score, 1),
            "declared_weight": round(self.weight, 3),
            "effective_weight": round(self.effective_weight, 3),
            "contribution": None if self.contribution is None else round(self.contribution, 2),
            "rationale": self.rationale,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class Scorecard:
    ticker: str
    period: str
    composite: float | None
    implied_band: str
    band_description: str
    coverage: float
    confidence: str
    factors: list[FactorScore] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    altman_z: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": self.period,
            "composite_score_0_100": None if self.composite is None else round(self.composite, 1),
            "implied_band": self.implied_band,
            "band_description": self.band_description,
            "factor_coverage": round(self.coverage, 3),
            "confidence": self.confidence,
            "factors": [f.to_dict() for f in self.factors],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "altman_z": self.altman_z,
            "warnings": self.warnings,
            "is_synthetic": self.is_synthetic,
            "disclaimer": DISCLAIMER,
        }


def score(period: Period) -> Scorecard:
    computed: list[tuple[Factor, ratio_mod.RatioResult, float | None]] = []
    available_weight = 0.0
    for factor in FACTORS:
        result = ratio_mod.compute(factor.ratio, period)
        raw = None if result.value is None else _interp(result.value, factor.anchors)
        if raw is not None:
            available_weight += factor.weight
        computed.append((factor, result, raw))

    total_weight = sum(f.weight for f in FACTORS)
    coverage_share = available_weight / total_weight if total_weight else 0.0

    factor_scores: list[FactorScore] = []
    composite = 0.0
    for factor, result, raw in computed:
        effective = (factor.weight / available_weight) if (raw is not None and available_weight) else 0.0
        contribution = None if raw is None else raw * effective
        if contribution is not None:
            composite += contribution
        factor_scores.append(FactorScore(
            ratio=factor.ratio,
            label=result.label,
            category=result.category,
            weight=factor.weight,
            effective_weight=effective,
            value=result.value,
            display=result.display(),
            score=raw,
            contribution=contribution,
            rationale=factor.rationale,
            unavailable_reason=result.unavailable_reason,
        ))

    card = Scorecard(
        ticker=period.ticker,
        period=period.key,
        composite=composite if available_weight else None,
        implied_band="n/a",
        band_description="",
        coverage=coverage_share,
        confidence=_confidence(coverage_share),
        factors=factor_scores,
        is_synthetic=period.is_synthetic,
    )

    if card.composite is not None:
        for threshold, band, description in _RATING_BANDS:
            if card.composite >= threshold:
                card.implied_band, card.band_description = band, description
                break

    ranked = [f for f in factor_scores if f.score is not None]
    ranked.sort(key=lambda f: f.score, reverse=True)
    card.strengths = [
        f"{f.label} at {f.display} (factor score {f.score:.0f}/100)"
        for f in ranked[:3] if f.score >= 60
    ]
    card.weaknesses = [
        f"{f.label} at {f.display} (factor score {f.score:.0f}/100)"
        for f in reversed(ranked[-3:]) if f.score < 55
    ]

    if coverage_share < 0.6:
        card.warnings.append(
            f"only {coverage_share:.0%} of factor weight could be computed; "
            "composite is directional at best"
        )
    missing = [f.ratio for f in factor_scores if f.score is None]
    if missing:
        card.warnings.append(f"factors excluded for missing data: {', '.join(missing)}")
    if period.is_synthetic:
        card.warnings.append("computed from synthetic demo data, not filed financials")

    card.altman_z = altman_z(period)
    return card


def _confidence(coverage_share: float) -> str:
    if coverage_share >= 0.85:
        return "high"
    if coverage_share >= 0.6:
        return "medium"
    return "low"


def altman_z(period: Period) -> dict[str, Any] | None:
    """Altman Z''-score (the non-manufacturer / emerging-market variant).

    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    """
    needed = ("working_capital", "retained_earnings", "ebit", "total_assets",
              "total_equity", "total_liabilities")
    missing = [c for c in needed if period.get(c) is None]
    if missing:
        return {"available": False, "missing_inputs": missing}

    ta = period.get("total_assets") or 0.0
    tl = period.get("total_liabilities") or 0.0
    if ta <= 0 or tl <= 0:
        return {"available": False, "missing_inputs": ["positive total_assets/total_liabilities"]}

    x1 = (period.get("working_capital") or 0.0) / ta
    x2 = (period.get("retained_earnings") or 0.0) / ta
    x3 = (period.get("ebit") or 0.0) / ta
    x4 = (period.get("total_equity") or 0.0) / tl
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    if z > 2.6:
        zone = "safe"
    elif z >= 1.1:
        zone = "grey"
    else:
        zone = "distress"
    return {
        "available": True,
        "model": "Altman Z''-score (non-manufacturer variant)",
        "score": round(z, 2),
        "zone": zone,
        "components": {
            "X1_working_capital_to_assets": round(x1, 4),
            "X2_retained_earnings_to_assets": round(x2, 4),
            "X3_ebit_to_assets": round(x3, 4),
            "X4_equity_to_liabilities": round(x4, 4),
        },
        "formula": "6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4",
        "interpretation": {"safe": "> 2.6", "grey": "1.1 - 2.6", "distress": "< 1.1"},
        "caveat": "Bankruptcy-prediction heuristic; weak for financial firms and asset-light models.",
    }


def compare_scorecards(cards: Sequence[Scorecard]) -> dict[str, Any]:
    """Side-by-side factor comparison for multi-issuer questions."""
    rows: list[dict[str, Any]] = []
    factor_names = [f.ratio for f in FACTORS]
    for name in factor_names:
        row: dict[str, Any] = {"factor": name}
        for card in cards:
            match = next((f for f in card.factors if f.ratio == name), None)
            row[card.ticker] = {
                "display": match.display if match else "n/a",
                "score": None if not match or match.score is None else round(match.score, 1),
            }
        rows.append(row)
    ranking = sorted(
        [c for c in cards if c.composite is not None], key=lambda c: -c.composite
    )
    return {
        "issuers": [
            {
                "ticker": c.ticker,
                "period": c.period,
                "composite": None if c.composite is None else round(c.composite, 1),
                "implied_band": c.implied_band,
                "confidence": c.confidence,
            }
            for c in cards
        ],
        "ranking": [c.ticker for c in ranking],
        "factor_table": rows,
        "disclaimer": DISCLAIMER,
    }
