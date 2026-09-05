"""Period assembly: raw facts -> comparable financial periods.

Two things make this non-trivial and worth doing in code rather than in a
prompt:

1. **Stock vs flow.** Balance-sheet items are point-in-time; income and cash
   flow items accumulate. Summing a balance across four quarters is a classic
   silent error, so `is_flow()` gates every aggregation.
2. **TTM.** Quarterly comparisons across companies with different fiscal
   calendars need trailing-twelve-month flows paired with the *latest* stock
   values. That pairing is done once, here, and every ratio consumes it.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from creditlens.db.models import Company, FinancialFact
from creditlens.finance import taxonomy
from creditlens.observability import get_logger

log = get_logger(__name__)

QUARTERS = ("Q1", "Q2", "Q3", "Q4")


@dataclass
class Provenance:
    """Where a single value came from. Attached to every number we surface."""

    concept: str
    source: str
    accession: str | None = None
    raw_concept: str | None = None
    filing_id: int | None = None
    is_synthetic: bool = False
    derived_from: list[str] = field(default_factory=list)
    formula: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "source": self.source,
            "accession": self.accession,
            "raw_concept": self.raw_concept,
            "filing_id": self.filing_id,
            "is_synthetic": self.is_synthetic,
            "derived_from": self.derived_from or None,
            "formula": self.formula,
        }


@dataclass
class Period:
    """All normalized facts for one company-period, plus derived aggregates."""

    ticker: str
    company_id: int
    fiscal_year: int
    fiscal_period: str  # FY, Q1..Q4, TTM
    period_end: dt.date
    values: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    is_synthetic: bool = False

    @property
    def key(self) -> str:
        return f"{self.fiscal_period}{self.fiscal_year}" if self.fiscal_period != "FY" else f"FY{self.fiscal_year}"

    @property
    def label(self) -> str:
        return f"{self.ticker} {self.key}"

    def get(self, concept: str) -> float | None:
        return self.values.get(concept)

    def require(self, *concepts: str) -> bool:
        return all(self.values.get(c) is not None for c in concepts)

    def sort_key(self) -> tuple:
        order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5, "TTM": 6}
        return (self.fiscal_year, order.get(self.fiscal_period, 0), self.period_end)

    def to_dict(self, concepts: Sequence[str] | None = None) -> dict[str, Any]:
        keys = list(concepts) if concepts else sorted(self.values)
        return {
            "ticker": self.ticker,
            "period": self.key,
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "period_end": self.period_end.isoformat(),
            "is_synthetic": self.is_synthetic,
            "values": {k: self.values[k] for k in keys if k in self.values},
            "provenance": {
                k: self.provenance[k].to_dict() for k in keys if k in self.provenance
            },
        }


def _put(period: Period, concept: str, value: float, prov: Provenance) -> None:
    period.values[concept] = value
    period.provenance[concept] = prov


def load_periods(
    session: Session,
    ticker: str,
    *,
    freq: str = "quarterly",
    limit: int | None = None,
) -> list[Period]:
    """Build `Period` objects for a company from stored facts.

    `freq` is "quarterly" (Q1..Q4), "annual" (FY), or "all".
    Returns ascending by period end; `limit` keeps the most recent N.
    """
    company = session.scalar(select(Company).where(Company.ticker == ticker.upper()))
    if company is None:
        return []

    stmt = select(FinancialFact).where(FinancialFact.company_id == company.id)
    if freq == "quarterly":
        stmt = stmt.where(FinancialFact.fiscal_period.in_(QUARTERS))
    elif freq == "annual":
        stmt = stmt.where(FinancialFact.fiscal_period == "FY")
    facts = list(session.scalars(stmt))

    buckets: dict[tuple[int, str], Period] = {}
    for fact in facts:
        key = (fact.fiscal_year, fact.fiscal_period)
        period = buckets.get(key)
        if period is None:
            period = Period(
                ticker=company.ticker,
                company_id=company.id,
                fiscal_year=fact.fiscal_year,
                fiscal_period=fact.fiscal_period,
                period_end=fact.period_end,
                is_synthetic=company.is_synthetic,
            )
            buckets[key] = period
        period.period_end = max(period.period_end, fact.period_end)
        period.is_synthetic = period.is_synthetic or fact.is_synthetic
        _put(
            period,
            fact.concept,
            fact.value,
            Provenance(
                concept=fact.concept,
                source=fact.source,
                accession=fact.accession,
                raw_concept=fact.raw_concept,
                filing_id=fact.filing_id,
                is_synthetic=fact.is_synthetic,
            ),
        )

    periods = sorted(buckets.values(), key=Period.sort_key)
    for period in periods:
        add_derived(period)
    if limit:
        periods = periods[-limit:]
    return periods


def add_derived(period: Period) -> Period:
    """Compute the aggregates credit analysis actually uses.

    Derived values are marked `source="derived"` with their formula, so the
    verifier and the UI can distinguish "the filer said this" from "we
    computed this".
    """

    def derive(name: str, formula: str, inputs: Sequence[str], fn) -> None:
        if name in period.values:
            return
        present = [c for c in inputs if period.values.get(c) is not None]
        required = [c for c in inputs if not c.startswith("?")]
        clean_required = [c.lstrip("?") for c in required]
        if not all(period.values.get(c) is not None for c in clean_required):
            return
        try:
            value = fn({c: period.values.get(c) or 0.0 for c in [i.lstrip("?") for i in inputs]})
        except (TypeError, ZeroDivisionError):
            return
        _put(
            period,
            name,
            value,
            Provenance(
                concept=name,
                source="derived",
                derived_from=present,
                formula=formula,
                is_synthetic=period.is_synthetic,
            ),
        )

    _derive_total_debt(period, derive)

    # D&A: prefer a combined tag; otherwise sum the separately reported parts.
    derive("depreciation_amortization",
           "depreciation_only + amortization_intangibles",
           ["depreciation_only", "?amortization_intangibles"],
           lambda v: v["depreciation_only"] + v["amortization_intangibles"])

    derive("net_debt", "total_debt - cash_and_equivalents - short_term_investments",
           ["total_debt", "cash_and_equivalents", "?short_term_investments"],
           lambda v: v["total_debt"] - v["cash_and_equivalents"] - v["short_term_investments"])
    derive("ebit", "operating_income", ["operating_income"], lambda v: v["operating_income"])
    derive("ebitda", "operating_income + depreciation_amortization",
           ["operating_income", "depreciation_amortization"],
           lambda v: v["operating_income"] + v["depreciation_amortization"])
    derive("free_cash_flow", "operating_cash_flow - capex",
           ["operating_cash_flow", "capex"],
           lambda v: v["operating_cash_flow"] - v["capex"])
    derive("working_capital", "current_assets - current_liabilities",
           ["current_assets", "current_liabilities"],
           lambda v: v["current_assets"] - v["current_liabilities"])
    derive("gross_profit", "revenue - cost_of_revenue",
           ["revenue", "cost_of_revenue"],
           lambda v: v["revenue"] - v["cost_of_revenue"])
    derive("tangible_equity", "total_equity - goodwill - intangible_assets",
           ["total_equity", "?goodwill", "?intangible_assets"],
           lambda v: v["total_equity"] - v["goodwill"] - v["intangible_assets"])
    derive("cash_and_investments", "cash_and_equivalents + short_term_investments",
           ["cash_and_equivalents", "?short_term_investments"],
           lambda v: v["cash_and_equivalents"] + v["short_term_investments"])
    return period


def _derive_total_debt(period: Period, derive) -> None:
    """Total debt, honouring how the filer actually tagged its debt.

    Three paths, in order of trust:
      1. a filer-reported combined total;
      2. short-term + long-term components;
      3. a long-term tag that already includes current maturities, used alone.
    """
    reported = period.values.get("total_debt_reported")
    if reported is not None:
        provenance = period.provenance.get("total_debt_reported")
        _put(period, "total_debt", reported, Provenance(
            concept="total_debt",
            source=provenance.source if provenance else "reported",
            accession=provenance.accession if provenance else None,
            raw_concept=provenance.raw_concept if provenance else None,
            filing_id=provenance.filing_id if provenance else None,
            formula="as reported by the filer",
            is_synthetic=period.is_synthetic,
        ))
        return

    short_term = period.values.get("short_term_debt")
    long_term = period.values.get("long_term_debt")
    if short_term is None and long_term is None:
        return

    long_term_tag = (
        period.provenance["long_term_debt"].raw_concept
        if "long_term_debt" in period.provenance else None
    )
    inclusive = long_term_tag in taxonomy.LONG_TERM_TAGS_INCLUDING_CURRENT

    if inclusive and long_term is not None:
        value = long_term
        formula = f"{long_term_tag} (already includes current maturities)"
        inputs = ["long_term_debt"]
    else:
        value = (short_term or 0.0) + (long_term or 0.0)
        formula = "short_term_debt + long_term_debt"
        inputs = [c for c in ("short_term_debt", "long_term_debt")
                  if period.values.get(c) is not None]

    _put(period, "total_debt", value, Provenance(
        concept="total_debt", source="derived", derived_from=inputs,
        formula=formula, is_synthetic=period.is_synthetic,
    ))


def build_ttm(periods: Sequence[Period]) -> Period | None:
    """Trailing-twelve-month period from the last four quarters.

    Flow concepts are summed; stock concepts are taken from the most recent
    quarter. Returns None if fewer than four consecutive quarters are present.
    """
    quarters = [p for p in periods if p.fiscal_period in QUARTERS]
    if len(quarters) < 4:
        return None
    window = sorted(quarters, key=Period.sort_key)[-4:]
    latest = window[-1]
    ttm = Period(
        ticker=latest.ticker,
        company_id=latest.company_id,
        fiscal_year=latest.fiscal_year,
        fiscal_period="TTM",
        period_end=latest.period_end,
        is_synthetic=any(p.is_synthetic for p in window),
    )
    concepts = {c for p in window for c in p.values}
    for concept in concepts:
        if taxonomy.is_flow(concept):
            vals = [p.values.get(concept) for p in window]
            if any(v is None for v in vals):
                continue
            total = sum(v for v in vals if v is not None)
            _put(ttm, concept, total, Provenance(
                concept=concept, source="derived-ttm",
                derived_from=[p.key for p in window],
                formula=f"sum({concept}) over {', '.join(p.key for p in window)}",
                is_synthetic=ttm.is_synthetic,
            ))
        else:
            value = latest.values.get(concept)
            if value is None:
                continue
            prov = latest.provenance.get(concept)
            _put(ttm, concept, value, Provenance(
                concept=concept,
                source=(prov.source if prov else "unknown"),
                accession=prov.accession if prov else None,
                raw_concept=prov.raw_concept if prov else None,
                filing_id=prov.filing_id if prov else None,
                is_synthetic=ttm.is_synthetic,
                formula=f"point-in-time as of {latest.key}",
            ))
    add_derived(ttm)
    return ttm


def select_period(periods: Sequence[Period], spec: str | None) -> Period | None:
    """Resolve a human period spec ("latest", "FY2024", "Q3-2025", "TTM")."""
    if not periods:
        return None
    ordered = sorted(periods, key=Period.sort_key)
    if not spec or spec.lower() in {"latest", "current", "most_recent"}:
        return ordered[-1]
    norm = spec.upper().replace("-", "").replace(" ", "").replace("_", "")
    # "TTM" and the rendered label "TTM2026" must both resolve to the trailing
    # twelve months, or a round trip through Period.key silently changes scope.
    if norm.startswith("TTM"):
        return build_ttm(ordered)
    for period in reversed(ordered):
        if period.key.upper() == norm:
            return period
    # "FY2024" style requested but only quarters present -> latest of that year
    digits = "".join(ch for ch in norm if ch.isdigit())
    if digits:
        year = int(digits[-4:]) if len(digits) >= 4 else None
        if year:
            same_year = [p for p in ordered if p.fiscal_year == year]
            if same_year:
                return same_year[-1]
    return None


def coverage(period: Period, concepts: Iterable[str]) -> dict[str, bool]:
    """Which requested concepts this period actually has - drives 'unknown' answers."""
    return {c: period.values.get(c) is not None for c in concepts}
