"""Period-over-period analysis and change attribution.

The "what caused X to change" question is answered here, arithmetically, not
by the model. For any ratio N/D the change decomposes *exactly*:

    N1/D1 - N0/D0  =  (N1-N0)/D1  -  N0*(D1-D0)/(D0*D1)
                      \\__________/     \\__________________/
                       numerator effect   denominator effect

so we can always say "margin fell 180bps: -240bps from cost growth, +60bps
from revenue growth" and have the two components sum to the total.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from creditlens.finance import ratios as ratio_mod
from creditlens.finance.statements import Period

IMPROVING = "improving"
DETERIORATING = "deteriorating"
STABLE = "stable"
MIXED = "mixed"


@dataclass
class Change:
    label: str
    from_period: str
    to_period: str
    from_value: float | None
    to_value: float | None
    absolute: float | None = None
    percent: float | None = None
    unit: str = ""
    higher_is_better: bool | None = None

    @property
    def direction(self) -> str:
        if self.absolute is None or self.higher_is_better is None:
            return STABLE if self.absolute in (None, 0) else MIXED
        if abs(self.absolute) < 1e-12:
            return STABLE
        improved = (self.absolute > 0) == self.higher_is_better
        return IMPROVING if improved else DETERIORATING

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "from_period": self.from_period,
            "to_period": self.to_period,
            "from_value": _r(self.from_value),
            "to_value": _r(self.to_value),
            "absolute_change": _r(self.absolute),
            "percent_change": _r(self.percent),
            "unit": self.unit,
            "direction": self.direction,
        }


@dataclass
class TrendResult:
    name: str
    label: str
    unit: str
    ticker: str
    points: list[dict[str, Any]] = field(default_factory=list)
    direction: str = STABLE
    slope_per_period: float | None = None
    r_squared: float | None = None
    total_change: float | None = None
    percent_change: float | None = None
    cagr_percent: float | None = None
    volatility: float | None = None
    consecutive_moves: int = 0
    higher_is_better: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "label": self.label,
            "ticker": self.ticker,
            "unit": self.unit,
            "series": self.points,
            "direction": self.direction,
            "slope_per_period": _r(self.slope_per_period),
            "r_squared": _r(self.r_squared),
            "total_change": _r(self.total_change),
            "percent_change": _r(self.percent_change),
            "cagr_percent": _r(self.cagr_percent),
            "volatility": _r(self.volatility),
            "consecutive_moves": self.consecutive_moves,
            "higher_is_better": self.higher_is_better,
            "notes": self.notes or None,
        }


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def pct_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def compare_concept(
    concept: str, older: Period, newer: Period, higher_is_better: bool | None = True
) -> Change:
    a, b = older.get(concept), newer.get(concept)
    return Change(
        label=concept,
        from_period=older.key,
        to_period=newer.key,
        from_value=a,
        to_value=b,
        absolute=None if a is None or b is None else b - a,
        percent=pct_change(a, b),
        unit="USD",
        higher_is_better=higher_is_better,
    )


def compare_ratio(name: str, older: Period, newer: Period) -> Change:
    defn = ratio_mod.REGISTRY[name]
    a = ratio_mod.compute(name, older)
    b = ratio_mod.compute(name, newer)
    return Change(
        label=defn.label,
        from_period=older.key,
        to_period=newer.key,
        from_value=a.value,
        to_value=b.value,
        absolute=None if a.value is None or b.value is None else b.value - a.value,
        percent=pct_change(a.value, b.value),
        unit=defn.unit,
        higher_is_better=defn.higher_is_better,
    )


def _ols_slope(ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope per period and R^2 for an evenly spaced series."""
    n = len(ys)
    if n < 2:
        return 0.0, 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 0.0 if syy == 0 else (sxy ** 2) / (sxx * syy)
    return slope, r2


def trend_for_ratio(name: str, periods: Sequence[Period]) -> TrendResult:
    defn = ratio_mod.REGISTRY[name]
    results = [ratio_mod.compute(name, p) for p in periods]
    usable = [(p, r) for p, r in zip(periods, results) if r.value is not None]

    out = TrendResult(
        name=name, label=defn.label, unit=defn.unit,
        ticker=periods[0].ticker if periods else "",
        higher_is_better=defn.higher_is_better,
        points=[
            {"period": p.key, "period_end": p.period_end.isoformat(),
             "value": _r(r.value), "display": r.display()}
            for p, r in zip(periods, results)
        ],
    )
    skipped = [r.period for r in results if r.value is None]
    if skipped:
        out.notes.append(f"no value for: {', '.join(skipped)}")
    if len(usable) < 2:
        out.notes.append("fewer than two comparable periods; no trend inferred")
        return out

    ys = [r.value for _, r in usable]
    out.total_change = ys[-1] - ys[0]
    out.percent_change = pct_change(ys[0], ys[-1])
    slope, r2 = _ols_slope(ys)
    out.slope_per_period, out.r_squared = slope, r2

    mean = sum(ys) / len(ys)
    if mean != 0:
        var = sum((y - mean) ** 2 for y in ys) / len(ys)
        out.volatility = math.sqrt(var) / abs(mean) * 100.0

    diffs = [b - a for a, b in zip(ys, ys[1:])]
    streak = 0
    for d in reversed(diffs):
        if d == 0 or (streak and (d > 0) != (diffs[-1] > 0)):
            break
        streak += 1
    out.consecutive_moves = streak * (1 if diffs and diffs[-1] > 0 else -1)

    # A move is "material" if it exceeds 5% of the series mean; below that we
    # call it stable rather than manufacturing a narrative from noise.
    threshold = max(abs(mean) * 0.05, 1e-9)
    if abs(out.total_change) < threshold:
        out.direction = STABLE
    elif defn.higher_is_better is None:
        out.direction = MIXED
    else:
        improved = (out.total_change > 0) == defn.higher_is_better
        out.direction = IMPROVING if improved else DETERIORATING
    if r2 < 0.4 and out.direction != STABLE:
        out.notes.append(f"non-monotonic path (R^2={r2:.2f}); endpoints drive the direction")
    return out


def cagr(first: float, last: float, years: float) -> float | None:
    if first <= 0 or last <= 0 or years <= 0:
        return None
    return ((last / first) ** (1 / years) - 1) * 100.0


@dataclass
class Attribution:
    metric: str
    from_period: str
    to_period: str
    total_change: float | None
    unit: str
    components: list[dict[str, Any]] = field(default_factory=list)
    method: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "from_period": self.from_period,
            "to_period": self.to_period,
            "total_change": _r(self.total_change),
            "unit": self.unit,
            "method": self.method,
            "components": self.components,
            "notes": self.notes or None,
        }


def attribute_ratio_change(name: str, older: Period, newer: Period) -> Attribution:
    """Split a ratio's change into numerator and denominator effects (exact)."""
    defn = ratio_mod.REGISTRY[name]
    a, b = ratio_mod.compute(name, older), ratio_mod.compute(name, newer)
    attribution = Attribution(
        metric=name, from_period=older.key, to_period=newer.key,
        total_change=None if a.value is None or b.value is None else b.value - a.value,
        unit=defn.unit,
        method="exact numerator/denominator decomposition",
    )
    if a.value is None or b.value is None:
        attribution.notes.append(a.unavailable_reason or b.unavailable_reason or "not computable")
        return attribution
    if len(defn.inputs) != 2:
        attribution.notes.append(
            "multi-input ratio; component attribution reported as input changes only"
        )
        attribution.components = [
            {
                "input": c,
                "from": _r(older.get(c)),
                "to": _r(newer.get(c)),
                "percent_change": _r(pct_change(older.get(c), newer.get(c))),
            }
            for c in defn.inputs
        ]
        return attribution

    num, den = defn.inputs
    n0, n1 = older.get(num), newer.get(num)
    d0, d1 = older.get(den), newer.get(den)
    if None in (n0, n1, d0, d1) or d0 == 0 or d1 == 0:
        attribution.notes.append("inputs unavailable for decomposition")
        return attribution

    scale = 100.0 if defn.unit == "%" else 1.0
    numerator_effect = (n1 - n0) / d1 * scale
    denominator_effect = -n0 * (d1 - d0) / (d0 * d1) * scale
    attribution.components = [
        {
            "driver": f"{num} change",
            "input": num,
            "from": _r(n0), "to": _r(n1),
            "percent_change": _r(pct_change(n0, n1)),
            "effect": _r(numerator_effect),
            "share_of_change": _r(_share(numerator_effect, attribution.total_change)),
        },
        {
            "driver": f"{den} change",
            "input": den,
            "from": _r(d0), "to": _r(d1),
            "percent_change": _r(pct_change(d0, d1)),
            "effect": _r(denominator_effect),
            "share_of_change": _r(_share(denominator_effect, attribution.total_change)),
        },
    ]
    residual = attribution.total_change - (numerator_effect + denominator_effect)
    if abs(residual) > max(abs(attribution.total_change) * 1e-6, 1e-9):
        attribution.notes.append(f"decomposition residual {residual:.6f} (expected ~0)")
    return attribution


def _share(part: float, whole: float | None) -> float | None:
    if whole in (None, 0):
        return None
    return part / whole * 100.0


MARGIN_COMPONENTS = (
    "cost_of_revenue", "rd_expense", "sga_expense", "depreciation_amortization",
)


def attribute_margin_change(older: Period, newer: Period, margin: str = "operating_margin") -> Attribution:
    """Explain an operating-margin move line by line, each as % of revenue.

    Components are expense lines expressed as a share of revenue; the change in
    each share is (minus) its contribution to margin. A residual line captures
    everything the filer did not tag separately.
    """
    a, b = ratio_mod.compute(margin, older), ratio_mod.compute(margin, newer)
    attribution = Attribution(
        metric=margin, from_period=older.key, to_period=newer.key,
        total_change=None if a.value is None or b.value is None else b.value - a.value,
        unit="pp", method="expense lines as a share of revenue",
    )
    r0, r1 = older.get("revenue"), newer.get("revenue")
    if a.value is None or b.value is None or not r0 or not r1:
        attribution.notes.append("revenue or margin unavailable in one of the periods")
        return attribution

    explained = 0.0
    for concept in MARGIN_COMPONENTS:
        c0, c1 = older.get(concept), newer.get(concept)
        if c0 is None or c1 is None:
            continue
        share0, share1 = c0 / r0 * 100.0, c1 / r1 * 100.0
        effect = -(share1 - share0)  # a rising expense share compresses margin
        explained += effect
        attribution.components.append({
            "driver": concept,
            "share_of_revenue_from": _r(share0),
            "share_of_revenue_to": _r(share1),
            "effect_pp": _r(effect),
            "share_of_change": _r(_share(effect, attribution.total_change)),
        })
    residual = attribution.total_change - explained
    attribution.components.append({
        "driver": "other / untagged items",
        "effect_pp": _r(residual),
        "share_of_change": _r(_share(residual, attribution.total_change)),
    })
    attribution.components.sort(key=lambda c: -abs(c.get("effect_pp") or 0))
    attribution.notes.append(
        f"revenue changed {pct_change(r0, r1):.1f}%" if pct_change(r0, r1) is not None else ""
    )
    attribution.notes = [n for n in attribution.notes if n]
    return attribution
