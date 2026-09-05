"""Credit ratio registry.

Every ratio is a declared object, not an inline expression. That buys three
things the LLM layer depends on:

* **Explainability** - each result carries the formula, the exact inputs it
  consumed, and the provenance of each input.
* **Guardrails** - a ratio with a meaningless denominator (negative EBITDA,
  zero interest expense) returns `None` plus a reason rather than a number that
  looks authoritative. Credit analysis is full of these traps.
* **Introspection** - the agent can list available ratios and their input
  requirements instead of inventing formulas.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from creditlens.finance.statements import Period

Category = str

ANNUAL_DAYS = 365.0
QUARTER_DAYS = 91.25


def period_days(period: Period) -> float:
    return QUARTER_DAYS if period.fiscal_period in {"Q1", "Q2", "Q3", "Q4"} else ANNUAL_DAYS


def annualization_factor(period: Period) -> float:
    """Multiplier that turns a period's flow into an annual-run-rate flow."""
    return 4.0 if period.fiscal_period in {"Q1", "Q2", "Q3", "Q4"} else 1.0


@dataclass
class RatioResult:
    name: str
    label: str
    category: Category
    value: float | None
    unit: str
    formula: str
    period: str
    ticker: str
    inputs: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    higher_is_better: bool | None = None
    warnings: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None
    is_synthetic: bool = False

    @property
    def ok(self) -> bool:
        return self.value is not None

    def display(self) -> str:
        if self.value is None:
            return "n/a"
        if self.unit == "%":
            return f"{self.value:.1f}%"
        if self.unit == "x":
            return f"{self.value:.2f}x"
        if self.unit == "days":
            return f"{self.value:.0f} days"
        return f"{self.value:,.0f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "value": None if self.value is None else round(self.value, 4),
            "display": self.display(),
            "unit": self.unit,
            "formula": self.formula,
            "period": self.period,
            "ticker": self.ticker,
            "inputs": {k: round(v, 2) for k, v in self.inputs.items()},
            "provenance": self.provenance,
            "higher_is_better": self.higher_is_better,
            "warnings": self.warnings or None,
            "unavailable_reason": self.unavailable_reason,
            "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True)
class RatioDef:
    name: str
    label: str
    category: Category
    unit: str
    inputs: tuple[str, ...]
    formula: str
    fn: Callable[[dict[str, float], Period], float]
    higher_is_better: bool | None = None
    description: str = ""
    #: concepts whose non-positivity makes the ratio meaningless
    positive_required: tuple[str, ...] = ()
    #: ratio is only meaningful on annualized flows
    annualize_flows: bool = False


def _pct(x: float) -> float:
    return x * 100.0


REGISTRY: dict[str, RatioDef] = {}


def register(defn: RatioDef) -> RatioDef:
    REGISTRY[defn.name] = defn
    return defn


# --------------------------------------------------------------------------
# Leverage
# --------------------------------------------------------------------------
register(RatioDef(
    "debt_to_ebitda", "Debt / EBITDA", "leverage", "x",
    ("total_debt", "ebitda"), "total_debt / EBITDA",
    lambda v, p: v["total_debt"] / (v["ebitda"] * annualization_factor(p)),
    higher_is_better=False, positive_required=("ebitda",), annualize_flows=True,
    description="Gross leverage. The single most-quoted credit metric; "
                "un-interpretable when EBITDA is negative.",
))
register(RatioDef(
    "net_debt_to_ebitda", "Net debt / EBITDA", "leverage", "x",
    ("net_debt", "ebitda"), "(total_debt - cash - ST investments) / EBITDA",
    lambda v, p: v["net_debt"] / (v["ebitda"] * annualization_factor(p)),
    higher_is_better=False, positive_required=("ebitda",), annualize_flows=True,
    description="Leverage net of liquid assets. Negative means net cash.",
))
register(RatioDef(
    "debt_to_equity", "Debt / Equity", "leverage", "x",
    ("total_debt", "total_equity"), "total_debt / total_equity",
    lambda v, p: v["total_debt"] / v["total_equity"],
    higher_is_better=False, positive_required=("total_equity",),
    description="Capital structure balance. Meaningless with negative book equity.",
))
register(RatioDef(
    "debt_to_assets", "Debt / Assets", "leverage", "%",
    ("total_debt", "total_assets"), "total_debt / total_assets",
    lambda v, p: _pct(v["total_debt"] / v["total_assets"]),
    higher_is_better=False, positive_required=("total_assets",),
))
register(RatioDef(
    "debt_to_capital", "Debt / Total capital", "leverage", "%",
    ("total_debt", "total_equity"), "total_debt / (total_debt + total_equity)",
    lambda v, p: _pct(v["total_debt"] / (v["total_debt"] + v["total_equity"])),
    higher_is_better=False,
    description="Preferred over D/E when equity is small: bounded 0-100%.",
))
register(RatioDef(
    "liabilities_to_assets", "Liabilities / Assets", "leverage", "%",
    ("total_liabilities", "total_assets"), "total_liabilities / total_assets",
    lambda v, p: _pct(v["total_liabilities"] / v["total_assets"]),
    higher_is_better=False, positive_required=("total_assets",),
))

# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------
register(RatioDef(
    "ebitda_interest_coverage", "EBITDA / Interest", "coverage", "x",
    ("ebitda", "interest_expense"), "EBITDA / interest expense",
    lambda v, p: v["ebitda"] / abs(v["interest_expense"]),
    higher_is_better=True, positive_required=("interest_expense",),
    description="Ability to service interest from operating cash generation.",
))
register(RatioDef(
    "ebit_interest_coverage", "EBIT / Interest", "coverage", "x",
    ("ebit", "interest_expense"), "EBIT / interest expense",
    lambda v, p: v["ebit"] / abs(v["interest_expense"]),
    higher_is_better=True, positive_required=("interest_expense",),
    description="Stricter than EBITDA coverage: charges the business for D&A.",
))
register(RatioDef(
    "fcf_interest_coverage", "FCF / Interest", "coverage", "x",
    ("free_cash_flow", "interest_expense"), "free cash flow / interest expense",
    lambda v, p: v["free_cash_flow"] / abs(v["interest_expense"]),
    higher_is_better=True, positive_required=("interest_expense",),
))
register(RatioDef(
    "average_cost_of_debt", "Average cost of debt", "coverage", "%",
    ("interest_expense", "total_debt"), "annualized interest expense / total debt",
    lambda v, p: _pct(abs(v["interest_expense"]) * annualization_factor(p) / v["total_debt"]),
    higher_is_better=False, positive_required=("total_debt",), annualize_flows=True,
    description="Implied blended coupon; a rising path signals refinancing stress.",
))

# --------------------------------------------------------------------------
# Liquidity
# --------------------------------------------------------------------------
register(RatioDef(
    "current_ratio", "Current ratio", "liquidity", "x",
    ("current_assets", "current_liabilities"), "current assets / current liabilities",
    lambda v, p: v["current_assets"] / v["current_liabilities"],
    higher_is_better=True, positive_required=("current_liabilities",),
))
register(RatioDef(
    "quick_ratio", "Quick ratio", "liquidity", "x",
    ("current_assets", "inventory", "current_liabilities"),
    "(current assets - inventory) / current liabilities",
    lambda v, p: (v["current_assets"] - v["inventory"]) / v["current_liabilities"],
    higher_is_better=True, positive_required=("current_liabilities",),
))
register(RatioDef(
    "cash_ratio", "Cash ratio", "liquidity", "x",
    ("cash_and_investments", "current_liabilities"),
    "(cash + ST investments) / current liabilities",
    lambda v, p: v["cash_and_investments"] / v["current_liabilities"],
    higher_is_better=True, positive_required=("current_liabilities",),
))
register(RatioDef(
    "cash_to_debt", "Cash / Debt", "liquidity", "%",
    ("cash_and_investments", "total_debt"), "(cash + ST investments) / total debt",
    lambda v, p: _pct(v["cash_and_investments"] / v["total_debt"]),
    higher_is_better=True, positive_required=("total_debt",),
))

# --------------------------------------------------------------------------
# Profitability
# --------------------------------------------------------------------------
register(RatioDef(
    "gross_margin", "Gross margin", "profitability", "%",
    ("gross_profit", "revenue"), "gross profit / revenue",
    lambda v, p: _pct(v["gross_profit"] / v["revenue"]),
    higher_is_better=True, positive_required=("revenue",),
))
register(RatioDef(
    "operating_margin", "Operating margin", "profitability", "%",
    ("operating_income", "revenue"), "operating income / revenue",
    lambda v, p: _pct(v["operating_income"] / v["revenue"]),
    higher_is_better=True, positive_required=("revenue",),
))
register(RatioDef(
    "ebitda_margin", "EBITDA margin", "profitability", "%",
    ("ebitda", "revenue"), "EBITDA / revenue",
    lambda v, p: _pct(v["ebitda"] / v["revenue"]),
    higher_is_better=True, positive_required=("revenue",),
))
register(RatioDef(
    "net_margin", "Net margin", "profitability", "%",
    ("net_income", "revenue"), "net income / revenue",
    lambda v, p: _pct(v["net_income"] / v["revenue"]),
    higher_is_better=True, positive_required=("revenue",),
))
register(RatioDef(
    "return_on_assets", "Return on assets", "profitability", "%",
    ("net_income", "total_assets"), "annualized net income / total assets",
    lambda v, p: _pct(v["net_income"] * annualization_factor(p) / v["total_assets"]),
    higher_is_better=True, positive_required=("total_assets",), annualize_flows=True,
))
register(RatioDef(
    "return_on_equity", "Return on equity", "profitability", "%",
    ("net_income", "total_equity"), "annualized net income / total equity",
    lambda v, p: _pct(v["net_income"] * annualization_factor(p) / v["total_equity"]),
    higher_is_better=True, positive_required=("total_equity",), annualize_flows=True,
))
register(RatioDef(
    "return_on_invested_capital", "Return on invested capital", "profitability", "%",
    ("ebit", "income_tax_expense", "pretax_income", "total_debt", "total_equity"),
    "annualized NOPAT / (total debt + total equity)",
    lambda v, p: _pct(
        v["ebit"] * (1 - _tax_rate(v)) * annualization_factor(p)
        / (v["total_debt"] + v["total_equity"])
    ),
    higher_is_better=True, annualize_flows=True,
    description="NOPAT over invested capital; tax rate implied from the filing.",
))


def _tax_rate(v: dict[str, float]) -> float:
    pretax = v.get("pretax_income") or 0.0
    tax = v.get("income_tax_expense") or 0.0
    if pretax <= 0:
        return 0.21  # statutory fallback when pre-tax income is negative
    rate = tax / pretax
    return min(max(rate, 0.0), 0.6)


# --------------------------------------------------------------------------
# Cash flow
# --------------------------------------------------------------------------
register(RatioDef(
    "ocf_margin", "Operating cash flow margin", "cash_flow", "%",
    ("operating_cash_flow", "revenue"), "operating cash flow / revenue",
    lambda v, p: _pct(v["operating_cash_flow"] / v["revenue"]),
    higher_is_better=True, positive_required=("revenue",),
))
register(RatioDef(
    "fcf_margin", "Free cash flow margin", "cash_flow", "%",
    ("free_cash_flow", "revenue"), "(operating cash flow - capex) / revenue",
    lambda v, p: _pct(v["free_cash_flow"] / v["revenue"]),
    higher_is_better=True, positive_required=("revenue",),
))
register(RatioDef(
    "fcf_conversion", "FCF conversion", "cash_flow", "%",
    ("free_cash_flow", "ebitda"), "free cash flow / EBITDA",
    lambda v, p: _pct(v["free_cash_flow"] / v["ebitda"]),
    higher_is_better=True, positive_required=("ebitda",),
    description="How much reported EBITDA turns into cash after capex.",
))
register(RatioDef(
    "cfo_to_debt", "Operating cash flow / Debt", "cash_flow", "%",
    ("operating_cash_flow", "total_debt"), "annualized operating cash flow / total debt",
    lambda v, p: _pct(v["operating_cash_flow"] * annualization_factor(p) / v["total_debt"]),
    higher_is_better=True, positive_required=("total_debt",), annualize_flows=True,
    description="A rating-agency staple: cash generation against the debt stack.",
))
register(RatioDef(
    "capex_to_revenue", "Capex intensity", "cash_flow", "%",
    ("capex", "revenue"), "capex / revenue",
    lambda v, p: _pct(v["capex"] / v["revenue"]),
    higher_is_better=None, positive_required=("revenue",),
))

# --------------------------------------------------------------------------
# Efficiency
# --------------------------------------------------------------------------
register(RatioDef(
    "asset_turnover", "Asset turnover", "efficiency", "x",
    ("revenue", "total_assets"), "annualized revenue / total assets",
    lambda v, p: v["revenue"] * annualization_factor(p) / v["total_assets"],
    higher_is_better=True, positive_required=("total_assets",), annualize_flows=True,
))
register(RatioDef(
    "days_sales_outstanding", "Days sales outstanding", "efficiency", "days",
    ("accounts_receivable", "revenue"), "receivables / revenue * days in period",
    lambda v, p: v["accounts_receivable"] / v["revenue"] * period_days(p),
    higher_is_better=False, positive_required=("revenue",),
))
register(RatioDef(
    "days_inventory", "Days inventory outstanding", "efficiency", "days",
    ("inventory", "cost_of_revenue"), "inventory / cost of revenue * days in period",
    lambda v, p: v["inventory"] / v["cost_of_revenue"] * period_days(p),
    higher_is_better=False, positive_required=("cost_of_revenue",),
))


# --------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------
def compute(name: str, period: Period) -> RatioResult:
    """Compute one ratio for one period, or explain why it cannot be computed."""
    defn = REGISTRY.get(name)
    if defn is None:
        raise KeyError(f"unknown ratio: {name}")

    result = RatioResult(
        name=defn.name, label=defn.label, category=defn.category, value=None,
        unit=defn.unit, formula=defn.formula, period=period.key, ticker=period.ticker,
        higher_is_better=defn.higher_is_better, is_synthetic=period.is_synthetic,
    )

    missing = [c for c in defn.inputs if period.values.get(c) is None]
    if missing:
        result.unavailable_reason = f"missing inputs: {', '.join(missing)}"
        return result

    values = {c: float(period.values[c]) for c in defn.inputs}
    result.inputs = values
    result.provenance = {
        c: period.provenance[c].to_dict() for c in defn.inputs if c in period.provenance
    }

    for concept in defn.positive_required:
        if values.get(concept, 0.0) <= 0:
            result.unavailable_reason = (
                f"{concept} is {values.get(concept, 0.0):,.0f}; the ratio is not "
                "economically meaningful with a non-positive denominator"
            )
            return result

    try:
        result.value = float(defn.fn(values, period))
    except ZeroDivisionError:
        result.unavailable_reason = "division by zero"
        return result

    if defn.annualize_flows and period.fiscal_period in {"Q1", "Q2", "Q3", "Q4"}:
        result.warnings.append(
            "quarterly flows annualized x4; seasonality is not adjusted for"
        )
    if period.is_synthetic:
        result.warnings.append("computed from synthetic demo data")
    return result


def compute_many(names: Iterable[str], period: Period) -> list[RatioResult]:
    return [compute(n, period) for n in names]


def compute_all(period: Period, categories: Sequence[str] | None = None) -> list[RatioResult]:
    names = [
        n for n, d in REGISTRY.items()
        if categories is None or d.category in categories
    ]
    return compute_many(names, period)


def catalog() -> list[dict[str, Any]]:
    """Machine-readable ratio catalog - exposed to the agent as a tool result."""
    return [
        {
            "name": d.name,
            "label": d.label,
            "category": d.category,
            "unit": d.unit,
            "formula": d.formula,
            "inputs": list(d.inputs),
            "higher_is_better": d.higher_is_better,
            "description": d.description,
        }
        for d in REGISTRY.values()
    ]


CATEGORIES = sorted({d.category for d in REGISTRY.values()})
