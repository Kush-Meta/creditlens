"""The agent's tool surface.

Nine tools, deliberately: eight read-only analytical tools plus the structured
final-answer tool. A larger surface measurably degrades tool selection, and
everything a credit question needs decomposes into these primitives.

Every tool is a thin, typed wrapper over deterministic code in
`creditlens.finance` and `creditlens.retrieval`. No tool asks the model for a
number, and every number a tool returns is registered in the evidence ledger
before the result goes back into the conversation.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from creditlens.agent.evidence import EvidenceLedger, NumericEvidence
from creditlens.config import get_settings
from creditlens.db.models import Company
from creditlens.finance import ratios as ratio_mod
from creditlens.finance import scorecard as scorecard_mod
from creditlens.finance import statements, trends
from creditlens.finance.statements import Period
from creditlens.observability import METRICS, Trace, get_logger
from creditlens.retrieval import hybrid

log = get_logger(__name__)

HEADLINE_CONCEPTS = (
    "revenue", "gross_profit", "operating_income", "ebitda", "net_income",
    "interest_expense", "depreciation_amortization",
    "total_debt", "net_debt", "cash_and_investments", "total_assets",
    "total_liabilities", "total_equity", "current_assets", "current_liabilities",
    "operating_cash_flow", "capex", "free_cash_flow",
)


class ToolError(Exception):
    """Raised for a bad tool argument; surfaced to the model as an error result."""


@dataclass
class ToolContext:
    session: Session
    ledger: EvidenceLedger
    trace: Trace
    period_cache: dict[tuple[str, str], list[Period]] | None = None

    def periods(self, ticker: str, freq: str = "quarterly") -> list[Period]:
        if self.period_cache is None:
            self.period_cache = {}
        key = (ticker.upper(), freq)
        if key not in self.period_cache:
            self.period_cache[key] = statements.load_periods(
                self.session, ticker, freq=freq
            )
        return self.period_cache[key]

    def resolve(self, ticker: str, period_spec: str | None, freq: str = "quarterly") -> Period:
        periods = self.periods(ticker, freq)
        if not periods:
            known = known_tickers(self.session)
            raise ToolError(
                f"no financial data for '{ticker}'. Available issuers: "
                f"{', '.join(known) if known else 'none ingested yet'}"
            )
        period = statements.select_period(periods, period_spec)
        if period is None:
            raise ToolError(
                f"period '{period_spec}' not available for {ticker}. "
                f"Available: {', '.join(p.key for p in periods[-8:])}"
            )
        return period


def known_tickers(session: Session) -> list[str]:
    return sorted(session.scalars(select(Company.ticker)))


# ---------------------------------------------------------------------------
# tool implementations
# ---------------------------------------------------------------------------
def tool_list_companies(ctx: ToolContext, **_: Any) -> dict[str, Any]:
    companies = list(ctx.session.scalars(select(Company).order_by(Company.ticker)))
    out = []
    for company in companies:
        periods = ctx.periods(company.ticker)
        out.append({
            "ticker": company.ticker,
            "name": company.name,
            "industry": company.industry,
            "cik": company.cik,
            "data_source": company.source,
            "is_synthetic": company.is_synthetic,
            "periods_available": [p.key for p in periods[-10:]],
            "latest_period": periods[-1].key if periods else None,
        })
    return {"companies": out, "count": len(out)}


def tool_get_financials(
    ctx: ToolContext,
    ticker: str,
    freq: str = "quarterly",
    periods: int = 5,
    concepts: Sequence[str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    series = ctx.periods(ticker, freq)
    if not series:
        raise ToolError(
            f"no financial data for '{ticker}'. Available issuers: "
            f"{', '.join(known_tickers(ctx.session)) or 'none ingested yet'}"
        )
    selected = series[-max(1, min(int(periods), 20)):]
    wanted = list(concepts) if concepts else list(HEADLINE_CONCEPTS)

    ttm = statements.build_ttm(series)
    rows: list[dict[str, Any]] = []
    for period in selected + ([ttm] if ttm else []):
        row: dict[str, Any] = {"period": period.key, "period_end": period.period_end.isoformat()}
        for concept in wanted:
            value = period.get(concept)
            if value is None:
                continue
            row[concept] = round(value, 2)
            provenance = period.provenance.get(concept)
            ctx.ledger.add_number(NumericEvidence(
                key=f"{ticker.upper()}:{period.key}:{concept}",
                label=f"{ticker.upper()} {period.key} {concept}",
                value=float(value), unit="USD",
                ticker=ticker.upper(), period=period.key,
                formula=provenance.formula if provenance else None,
                source=provenance.source if provenance else "reported",
                tool="get_financials",
                provenance=provenance.to_dict() if provenance else {},
                is_synthetic=period.is_synthetic,
            ))
        rows.append(row)

    growth = _growth_table(selected, "revenue")
    for entry in growth:
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{entry['from']}->{entry['to']}:revenue_growth",
            label=f"{ticker.upper()} revenue growth {entry['from']} to {entry['to']}",
            value=float(entry["percent_change"]), unit="%",
            ticker=ticker.upper(), period=f"{entry['from']}->{entry['to']}",
            formula="(revenue_t / revenue_t-1 - 1) * 100",
            source="deterministic-calculation", tool="get_financials",
            is_synthetic=any(p.is_synthetic for p in selected),
        ))
    return {
        "ticker": ticker.upper(),
        "frequency": freq,
        "units": "as reported (USD)",
        "periods": rows,
        "revenue_growth": growth,
        "note": (
            "Values sourced from XBRL facts or derived per the recorded formula. "
            "TTM row sums flow items over the last four quarters and takes "
            "balance-sheet items from the most recent quarter."
        ),
        "is_synthetic": any(p.is_synthetic for p in selected),
    }


def _growth_table(periods: Sequence[Period], concept: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for older, newer in zip(periods, periods[1:]):
        change = trends.pct_change(older.get(concept), newer.get(concept))
        if change is None:
            continue
        out.append({
            "from": older.key, "to": newer.key,
            "percent_change": round(change, 2),
        })
    return out


def tool_compute_ratios(
    ctx: ToolContext,
    ticker: str,
    period: str | None = None,
    ratios: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    resolved = ctx.resolve(ticker, period)
    if ratios:
        unknown = [r for r in ratios if r not in ratio_mod.REGISTRY]
        if unknown:
            raise ToolError(
                f"unknown ratio(s): {', '.join(unknown)}. "
                f"Valid names: {', '.join(sorted(ratio_mod.REGISTRY))}"
            )
        results = ratio_mod.compute_many(ratios, resolved)
    else:
        results = ratio_mod.compute_all(resolved, categories=list(categories) if categories else None)

    for result in results:
        if result.value is None:
            continue
        ctx.ledger.add_number(NumericEvidence(
            key=f"{result.ticker}:{result.period}:{result.name}",
            label=f"{result.ticker} {result.period} {result.label}",
            value=float(result.value), unit=result.unit,
            ticker=result.ticker, period=result.period,
            formula=result.formula, source="deterministic-calculation",
            tool="compute_ratios", provenance=result.provenance,
            is_synthetic=result.is_synthetic,
        ))

    return {
        "ticker": resolved.ticker,
        "period": resolved.key,
        "period_end": resolved.period_end.isoformat(),
        "ratios": [r.to_dict() for r in results],
        "unavailable": [
            {"name": r.name, "reason": r.unavailable_reason}
            for r in results if r.value is None
        ],
        "is_synthetic": resolved.is_synthetic,
    }


def tool_compare_periods(
    ctx: ToolContext,
    ticker: str,
    metric: str,
    from_period: str | None = None,
    to_period: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    series = ctx.periods(ticker)
    if len(series) < 2:
        raise ToolError(f"need at least two periods for {ticker}; have {len(series)}")
    newer = ctx.resolve(ticker, to_period)
    older = ctx.resolve(ticker, from_period) if from_period else _prior(series, newer)
    if older.key == newer.key:
        raise ToolError("from_period and to_period resolved to the same period")

    is_ratio = metric in ratio_mod.REGISTRY
    change = (
        trends.compare_ratio(metric, older, newer)
        if is_ratio
        else trends.compare_concept(metric, older, newer)
    )
    if not is_ratio and older.get(metric) is None and newer.get(metric) is None:
        raise ToolError(
            f"unknown metric '{metric}'. Use a ratio name "
            f"({', '.join(sorted(ratio_mod.REGISTRY)[:6])}, ...) or a financial "
            f"concept ({', '.join(HEADLINE_CONCEPTS[:6])}, ...)"
        )

    attribution = (
        trends.attribute_ratio_change(metric, older, newer) if is_ratio else None
    )
    margin_bridge = (
        trends.attribute_margin_change(older, newer, metric)
        if metric in {"operating_margin", "gross_margin", "ebitda_margin", "net_margin"}
        else None
    )

    if change.absolute is not None:
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{older.key}->{newer.key}:{metric}:change",
            label=f"{ticker.upper()} {metric} change {older.key} to {newer.key}",
            value=float(change.absolute), unit=change.unit,
            ticker=ticker.upper(), period=f"{older.key}->{newer.key}",
            formula=f"{newer.key} minus {older.key}",
            source="deterministic-calculation", tool="compare_periods",
        ))
    if change.percent is not None:
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{older.key}->{newer.key}:{metric}:pct",
            label=f"{ticker.upper()} {metric} percent change",
            value=float(change.percent), unit="%",
            ticker=ticker.upper(), period=f"{older.key}->{newer.key}",
            source="deterministic-calculation", tool="compare_periods",
        ))
    for period in (older, newer):
        value = (
            ratio_mod.compute(metric, period).value if is_ratio else period.get(metric)
        )
        if value is not None:
            ctx.ledger.add_number(NumericEvidence(
                key=f"{ticker.upper()}:{period.key}:{metric}",
                label=f"{ticker.upper()} {period.key} {metric}",
                value=float(value), unit=change.unit,
                ticker=ticker.upper(), period=period.key,
                source="deterministic-calculation", tool="compare_periods",
            ))

    payload: dict[str, Any] = {
        "ticker": ticker.upper(),
        "metric": metric,
        "change": change.to_dict(),
        "is_synthetic": older.is_synthetic or newer.is_synthetic,
    }
    if attribution:
        payload["attribution"] = attribution.to_dict()
    if margin_bridge:
        payload["margin_bridge"] = margin_bridge.to_dict()
    return payload


def _prior(series: Sequence[Period], period: Period) -> Period:
    ordered = sorted(series, key=Period.sort_key)
    for i, candidate in enumerate(ordered):
        if candidate.key == period.key and i > 0:
            return ordered[i - 1]
    return ordered[0]


def tool_metric_trend(
    ctx: ToolContext,
    ticker: str,
    metric: str,
    freq: str = "quarterly",
    lookback: int = 6,
    **_: Any,
) -> dict[str, Any]:
    if metric not in ratio_mod.REGISTRY:
        return _concept_trend(ctx, ticker, metric, freq, lookback)
    series = ctx.periods(ticker, freq)
    if not series:
        raise ToolError(f"no financial data for '{ticker}'")
    window = series[-max(2, min(int(lookback), 24)):]
    result = trends.trend_for_ratio(metric, window)

    for point in result.points:
        if point["value"] is None:
            continue
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{point['period']}:{metric}",
            label=f"{ticker.upper()} {point['period']} {result.label}",
            value=float(point["value"]), unit=result.unit,
            ticker=ticker.upper(), period=point["period"],
            source="deterministic-calculation", tool="metric_trend",
            is_synthetic=any(p.is_synthetic for p in window),
        ))
    if result.total_change is not None:
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{metric}:total_change",
            label=f"{ticker.upper()} {result.label} total change over window",
            value=float(result.total_change), unit=result.unit,
            ticker=ticker.upper(), period=f"{window[0].key}->{window[-1].key}",
            source="deterministic-calculation", tool="metric_trend",
        ))
    for field_name, unit in (("percent_change", "%"), ("cagr_percent", "%"),
                             ("volatility", "%"), ("slope_per_period", result.unit)):
        value = getattr(result, field_name)
        if value is None:
            continue
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{metric}:{field_name}",
            label=f"{ticker.upper()} {result.label} {field_name.replace('_', ' ')}",
            value=float(value), unit=unit,
            ticker=ticker.upper(), period=f"{window[0].key}->{window[-1].key}",
            source="deterministic-calculation", tool="metric_trend",
        ))
    return {"ticker": ticker.upper(), "frequency": freq, **result.to_dict()}


def _concept_trend(
    ctx: ToolContext, ticker: str, concept: str, freq: str, lookback: int
) -> dict[str, Any]:
    series = ctx.periods(ticker, freq)
    if not series:
        raise ToolError(f"no financial data for '{ticker}'")
    window = series[-max(2, min(int(lookback), 24)):]
    points = [
        {"period": p.key, "period_end": p.period_end.isoformat(),
         "value": None if p.get(concept) is None else round(p.get(concept), 2)}
        for p in window
    ]
    values = [p["value"] for p in points if p["value"] is not None]
    if len(values) < 2:
        raise ToolError(
            f"'{concept}' is not a known ratio and has fewer than two reported "
            f"values for {ticker}"
        )
    for point in points:
        if point["value"] is None:
            continue
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{point['period']}:{concept}",
            label=f"{ticker.upper()} {point['period']} {concept}",
            value=float(point["value"]), unit="USD",
            ticker=ticker.upper(), period=point["period"],
            source="reported", tool="metric_trend",
            is_synthetic=any(p.is_synthetic for p in window),
        ))
    change = trends.pct_change(values[0], values[-1])
    if change is not None:
        ctx.ledger.add_number(NumericEvidence(
            key=f"{ticker.upper()}:{concept}:percent_change",
            label=f"{ticker.upper()} {concept} percent change over window",
            value=float(change), unit="%",
            ticker=ticker.upper(), period=f"{points[0]['period']}->{points[-1]['period']}",
            source="deterministic-calculation", tool="metric_trend",
        ))
    return {
        "ticker": ticker.upper(), "metric": concept, "unit": "USD",
        "frequency": freq, "series": points,
        "total_change": round(values[-1] - values[0], 2),
        "percent_change": None if change is None else round(change, 2),
        "direction": "increasing" if values[-1] > values[0] else "decreasing",
        "note": "raw financial concept; direction is not a credit judgement",
    }


def tool_credit_scorecard(
    ctx: ToolContext, ticker: str, period: str | None = None, **_: Any
) -> dict[str, Any]:
    resolved = ctx.resolve(ticker, period)
    card = scorecard_mod.score(resolved)
    if card.composite is not None:
        ctx.ledger.add_number(NumericEvidence(
            key=f"{card.ticker}:{card.period}:composite_score",
            label=f"{card.ticker} {card.period} composite credit score",
            value=float(card.composite), unit="score",
            ticker=card.ticker, period=card.period,
            source="deterministic-scorecard", tool="credit_scorecard",
            is_synthetic=card.is_synthetic,
        ))
    ctx.ledger.add_number(NumericEvidence(
        key=f"{card.ticker}:{card.period}:factor_coverage",
        label=f"{card.ticker} {card.period} scorecard factor coverage",
        value=float(card.coverage) * 100.0, unit="%",
        ticker=card.ticker, period=card.period,
        formula="weight of computable factors / total declared weight",
        source="deterministic-scorecard", tool="credit_scorecard",
        is_synthetic=card.is_synthetic,
    ))
    for factor in card.factors:
        if factor.value is None:
            continue
        ctx.ledger.add_number(NumericEvidence(
            key=f"{card.ticker}:{card.period}:{factor.ratio}",
            label=f"{card.ticker} {card.period} {factor.label}",
            value=float(factor.value), unit=_unit_of(factor.ratio),
            ticker=card.ticker, period=card.period,
            source="deterministic-calculation", tool="credit_scorecard",
            is_synthetic=card.is_synthetic,
        ))
    if card.altman_z and card.altman_z.get("available"):
        ctx.ledger.add_number(NumericEvidence(
            key=f"{card.ticker}:{card.period}:altman_z",
            label=f"{card.ticker} {card.period} Altman Z''-score",
            value=float(card.altman_z["score"]), unit="score",
            ticker=card.ticker, period=card.period,
            formula=card.altman_z["formula"],
            source="deterministic-scorecard", tool="credit_scorecard",
            is_synthetic=card.is_synthetic,
        ))
    return card.to_dict()


def _unit_of(ratio_name: str) -> str:
    defn = ratio_mod.REGISTRY.get(ratio_name)
    return defn.unit if defn else ""


def tool_compare_companies(
    ctx: ToolContext,
    tickers: Sequence[str],
    period: str | None = None,
    ratios: Sequence[str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if not tickers or len(tickers) < 2:
        raise ToolError("compare_companies needs at least two tickers")
    names = list(ratios) if ratios else [
        "debt_to_ebitda", "net_debt_to_ebitda", "ebitda_interest_coverage",
        "current_ratio", "cash_to_debt", "ebitda_margin", "fcf_margin", "cfo_to_debt",
    ]
    unknown = [r for r in names if r not in ratio_mod.REGISTRY]
    if unknown:
        raise ToolError(f"unknown ratio(s): {', '.join(unknown)}")

    table: list[dict[str, Any]] = []
    cards: list[scorecard_mod.Scorecard] = []
    resolved_periods: dict[str, str] = {}
    for ticker in tickers:
        resolved = ctx.resolve(ticker, period)
        resolved_periods[resolved.ticker] = resolved.key
        cards.append(scorecard_mod.score(resolved))
        for result in ratio_mod.compute_many(names, resolved):
            if result.value is not None:
                ctx.ledger.add_number(NumericEvidence(
                    key=f"{result.ticker}:{result.period}:{result.name}",
                    label=f"{result.ticker} {result.period} {result.label}",
                    value=float(result.value), unit=result.unit,
                    ticker=result.ticker, period=result.period,
                    formula=result.formula, source="deterministic-calculation",
                    tool="compare_companies", is_synthetic=result.is_synthetic,
                ))
            table.append({
                "ticker": result.ticker, "period": result.period,
                "ratio": result.name, "label": result.label,
                "value": None if result.value is None else round(result.value, 4),
                "display": result.display(), "unit": result.unit,
                "higher_is_better": result.higher_is_better,
                "unavailable_reason": result.unavailable_reason,
            })

    by_ratio: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        by_ratio.setdefault(row["ratio"], []).append(row)
    winners: dict[str, str | None] = {}
    for name, rows in by_ratio.items():
        scored = [r for r in rows if r["value"] is not None]
        polarity = ratio_mod.REGISTRY[name].higher_is_better
        if not scored or polarity is None:
            winners[name] = None
            continue
        best = max(scored, key=lambda r: r["value"]) if polarity else min(
            scored, key=lambda r: r["value"]
        )
        winners[name] = best["ticker"]

    return {
        "periods_compared": resolved_periods,
        "comparison": table,
        "stronger_on_each_ratio": winners,
        "scorecards": scorecard_mod.compare_scorecards(cards),
        "caveat": (
            "Periods are aligned by label, not by fiscal calendar; issuers with "
            "different fiscal year ends are not perfectly comparable."
        ),
    }


def tool_search_filings(
    ctx: ToolContext,
    query: str,
    tickers: Sequence[str] | None = None,
    form_types: Sequence[str] | None = None,
    items: Sequence[str] | None = None,
    fiscal_years: Sequence[int] | None = None,
    top_k: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    settings = get_settings()
    result = hybrid.search(
        ctx.session, query,
        tickers=tickers, form_types=form_types, item_codes=items,
        fiscal_years=fiscal_years,
        top_k=min(int(top_k or settings.retrieval_top_k), 15),
    )
    passages = []
    for hit in result.chunks:
        evidence = ctx.ledger.add_passage(hit)
        passages.append({
            "citation": evidence.label,
            "ticker": evidence.ticker,
            "source": f"{evidence.form_type} {evidence.period}",
            "item": evidence.item,
            "section": evidence.section,
            "filing_date": evidence.filing_date,
            "relevance": round(hit.score, 5),
            "text": hit.record.text,
        })
    return {
        "query": result.query,
        "expanded_query": result.expanded_query,
        "results_found": len(passages),
        "passages": passages,
        "retrieval_diagnostics": result.diagnostics,
        "instruction": (
            "Cite these passages by their citation label, e.g. [C1], for every "
            "qualitative claim drawn from them."
        ),
    }


def tool_data_coverage(ctx: ToolContext, ticker: str, **_: Any) -> dict[str, Any]:
    """What the corpus actually holds for an issuer, and what that blocks.

    Real XBRL coverage is uneven: some filers tag debt only at segment level
    through custom extensions, some never tag interest expense separately. A
    credit tool must be able to say precisely which metric it cannot compute
    and why, instead of returning silence or a guess.
    """
    series = ctx.periods(ticker, "quarterly")
    annual = ctx.periods(ticker, "annual")
    if not series and not annual:
        raise ToolError(
            f"no financial data for '{ticker}'. Available issuers: "
            f"{', '.join(known_tickers(ctx.session)) or 'none ingested yet'}"
        )
    reference = statements.build_ttm(series) or (annual[-1] if annual else series[-1])

    available, missing = [], []
    for concept in HEADLINE_CONCEPTS:
        (available if reference.get(concept) is not None else missing).append(concept)

    blocked: list[dict[str, Any]] = []
    for name in sorted(ratio_mod.REGISTRY):
        result = ratio_mod.compute(name, reference)
        if result.value is None:
            blocked.append({"ratio": name, "reason": result.unavailable_reason})

    company = ctx.session.scalar(select(Company).where(Company.ticker == ticker.upper()))
    return {
        "ticker": ticker.upper(),
        "reference_period": reference.key,
        "data_source": company.source if company else "unknown",
        "is_synthetic": reference.is_synthetic,
        "quarterly_periods": [p.key for p in series],
        "annual_periods": [p.key for p in annual],
        "concepts_available": available,
        "concepts_missing": missing,
        "ratios_available": len(ratio_mod.REGISTRY) - len(blocked),
        "ratios_blocked": blocked,
        "guidance": (
            "Concepts listed as missing are absent from the issuer's XBRL "
            "facts - typically because the filer tags them only at segment "
            "level through custom taxonomy extensions, or folds them into a "
            "combined line. Report these as unavailable; do not estimate them "
            "or substitute a different metric without saying so."
        ),
    }


def tool_submit_analysis(ctx: ToolContext, **payload: Any) -> dict[str, Any]:
    """Terminal tool: the model's structured final answer."""
    return {"received": True, **payload}


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
_TICKER = {"type": "string", "description": "Issuer ticker symbol, e.g. MSFT."}
_PERIOD = {
    "type": "string",
    "description": (
        "Period label: 'latest' (default), 'TTM', a fiscal year like 'FY2024', "
        "or a quarter like 'Q2-2025'."
    ),
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_companies",
        "description": (
            "List every issuer available in the corpus with the periods held for "
            "each. Call this first whenever the question names a company you are "
            "not sure has been ingested."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_financials",
        "description": (
            "Reported and derived financial line items for an issuer across "
            "periods, including a trailing-twelve-month row. Use this for raw "
            "amounts (revenue, debt, EBITDA, cash flow) and revenue growth. "
            "Never compute these numbers yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": _TICKER,
                "freq": {"type": "string", "enum": ["quarterly", "annual", "all"],
                          "description": "Reporting frequency. Default quarterly."},
                "periods": {"type": "integer", "minimum": 1, "maximum": 20,
                             "description": "How many recent periods to return. Default 5."},
                "concepts": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Optional subset of concepts, e.g. ['revenue','total_debt',"
                        "'ebitda','free_cash_flow']. Omit for the standard headline set."
                    ),
                },
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compute_ratios",
        "description": (
            "Compute credit ratios for one issuer-period from the deterministic "
            "ratio engine. Returns the value, the formula, the exact inputs used "
            "and their provenance, or an explicit reason the ratio is not "
            "meaningful. This is the only acceptable source for a ratio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": _TICKER,
                "period": _PERIOD,
                "ratios": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Ratio names, e.g. ['debt_to_ebitda','net_debt_to_ebitda',"
                        "'ebitda_interest_coverage','current_ratio','fcf_margin']."
                    ),
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": [
                        "leverage", "coverage", "liquidity", "profitability",
                        "cash_flow", "efficiency",
                    ]},
                    "description": "Compute every ratio in these categories instead.",
                },
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_periods",
        "description": (
            "Compare one metric between two periods and decompose the change. "
            "For ratios it returns an exact numerator/denominator attribution; for "
            "margins it also returns a line-by-line bridge. Use this to answer "
            "'what caused X to change'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": _TICKER,
                "metric": {
                    "type": "string",
                    "description": (
                        "A ratio name (e.g. 'operating_margin', 'debt_to_ebitda') or "
                        "a financial concept (e.g. 'revenue', 'total_debt')."
                    ),
                },
                "from_period": _PERIOD,
                "to_period": _PERIOD,
            },
            "required": ["ticker", "metric"],
            "additionalProperties": False,
        },
    },
    {
        "name": "metric_trend",
        "description": (
            "Trend of one metric over consecutive periods: the series, an "
            "ordinary-least-squares slope, R-squared, total and percent change, "
            "volatility and a direction classified against the metric's credit "
            "polarity. Use this for 'over the last N quarters/years' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": _TICKER,
                "metric": {"type": "string", "description": "Ratio name or financial concept."},
                "freq": {"type": "string", "enum": ["quarterly", "annual"]},
                "lookback": {"type": "integer", "minimum": 2, "maximum": 24,
                              "description": "Number of periods. Default 6."},
            },
            "required": ["ticker", "metric"],
            "additionalProperties": False,
        },
    },
    {
        "name": "credit_scorecard",
        "description": (
            "Run the internal weighted credit scorecard for one issuer-period. "
            "Returns a 0-100 composite, an implied rating band, per-factor scores "
            "and weights, strengths, weaknesses and an Altman Z''-score. It is a "
            "heuristic model, not a credit rating - present it as such."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": _TICKER, "period": _PERIOD},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_companies",
        "description": (
            "Side-by-side ratio and scorecard comparison for two or more issuers "
            "in the same period. Use this for any 'compare A and B' question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "period": _PERIOD,
                "ratios": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tickers"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_filings",
        "description": (
            "Hybrid semantic + keyword search over ingested 10-K/10-Q text. Use it "
            "for management commentary, risk factors, liquidity discussion and "
            "anything qualitative. Filter by item to target a section: '1A' Risk "
            "Factors, '7' MD&A, '7A' market risk, '1' Business, '2' quarterly MD&A. "
            "Every qualitative claim in your answer must cite a passage returned here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "tickers": {"type": "array", "items": {"type": "string"}},
                "form_types": {"type": "array", "items": {"type": "string", "enum": ["10-K", "10-Q", "8-K"]}},
                "items": {"type": "array", "items": {"type": "string"},
                           "description": "Item codes such as ['1A'] or ['7','7A']."},
                "fiscal_years": {"type": "array", "items": {"type": "integer"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 15},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "data_coverage",
        "description": (
            "Report which financial concepts exist for an issuer and which "
            "ratios are consequently not computable, with the reason for each. "
            "Call this when a metric you expected comes back unavailable, so "
            "you can explain the gap precisely instead of guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": _TICKER},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_analysis",
        "description": (
            "Submit the final structured analysis. Call this exactly once, as your "
            "last action, after gathering evidence. Every number in it must have "
            "come from a tool result, and every qualitative claim must carry a "
            "citation label."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Direct answer to the question in 2-5 sentences. Lead with the "
                        "conclusion. Cite passages inline as [C1]."
                    ),
                },
                "credit_direction": {
                    "type": "string",
                    "enum": ["improving", "stable", "deteriorating", "mixed", "not_applicable"],
                    "description": (
                        "Overall direction of credit quality implied by the evidence. "
                        "Use 'not_applicable' for questions that are not about credit direction."
                    ),
                },
                "key_metrics": {
                    "type": "array",
                    "description": "The metrics that drive the conclusion. Values must match tool output exactly.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string", "description": "Formatted value, e.g. '3.42x' or '18.6%'."},
                            "period": {"type": "string"},
                            "ticker": {"type": "string"},
                            "commentary": {"type": "string"},
                        },
                        "required": ["name", "value", "period"],
                        "additionalProperties": False,
                    },
                },
                "positive_factors": {"type": "array", "items": {"type": "string"},
                                      "description": "Credit strengths, each with a citation where qualitative."},
                "risk_factors": {"type": "array", "items": {"type": "string"},
                                  "description": "Credit risks, each with a citation where qualitative."},
                "reasoning": {
                    "type": "string",
                    "description": (
                        "The analytical chain: which metrics moved, by how much, what "
                        "drove them, and what the filings say about why. 1-4 paragraphs."
                    ),
                },
                "citations": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Every citation label used, e.g. ['C1','C4'].",
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "caveats": {"type": "array", "items": {"type": "string"},
                             "description": "Data gaps, comparability issues, or limits on the conclusion."},
            },
            "required": ["answer", "credit_direction", "key_metrics", "reasoning", "confidence"],
            "additionalProperties": False,
        },
    },
]

TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "list_companies": tool_list_companies,
    "get_financials": tool_get_financials,
    "compute_ratios": tool_compute_ratios,
    "compare_periods": tool_compare_periods,
    "metric_trend": tool_metric_trend,
    "credit_scorecard": tool_credit_scorecard,
    "compare_companies": tool_compare_companies,
    "search_filings": tool_search_filings,
    "data_coverage": tool_data_coverage,
    "submit_analysis": tool_submit_analysis,
}

TERMINAL_TOOL = "submit_analysis"
ANALYTICAL_TOOLS = [s for s in TOOL_SCHEMAS if s["name"] != TERMINAL_TOOL]


def execute(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Run a tool. Returns (result, is_error). Errors are returned, not raised.

    A failed tool call is recoverable information for the model - the error
    text names valid alternatives - so it goes back as a `tool_result` with
    `is_error`, never as an exception that kills the turn.
    """
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        METRICS.inc("creditlens_tool_calls_total", tool=name, outcome="unknown_tool")
        return {"error": f"unknown tool '{name}'",
                "available_tools": sorted(TOOL_IMPLS)}, True

    started = dt.datetime.now(dt.UTC)
    with ctx.trace.span(f"tool.{name}", kind="tool", arguments=_short(arguments)) as span:
        try:
            result = impl(ctx, **arguments)
            outcome = "ok"
        except ToolError as exc:
            result, outcome = {"error": str(exc)}, "tool_error"
        except TypeError as exc:
            result, outcome = (
                {"error": f"invalid arguments for {name}: {exc}"}, "bad_arguments"
            )
        except Exception as exc:
            log.exception("tool crashed", extra={"tool": name})
            result, outcome = (
                {"error": f"{type(exc).__name__}: {exc}"}, "exception"
            )
        elapsed_ms = (dt.datetime.now(dt.UTC) - started).total_seconds() * 1000
        span.set(outcome=outcome, duration_ms=round(elapsed_ms, 2))

    METRICS.inc("creditlens_tool_calls_total", tool=name, outcome=outcome)
    METRICS.observe("creditlens_tool_latency_ms", elapsed_ms, tool=name)
    return result, outcome != "ok"


def _short(arguments: dict[str, Any]) -> str:
    text = str(arguments)
    return text if len(text) <= 300 else text[:297] + "..."
