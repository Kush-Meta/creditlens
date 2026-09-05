"""Offline demo corpus: four fictional issuers with articulated financials.

Why fictional issuers rather than real tickers with made-up numbers: a demo
corpus that attributes invented financials to a real company produces
documents that look exactly like fabricated filings. Everything here is
clearly fictional, every row is stamped `is_synthetic=True`, and the API
refuses to hide that flag. Real companies are analysed by ingesting their
actual EDGAR filings (`creditlens ingest --ticker F`).

The generated statements *articulate*: operating income is built from the
revenue and expense lines, net income rolls into equity, cash rolls forward
through free cash flow, financing and buybacks, and the balance sheet is
forced to balance through a single explicit plug. That matters because the
ratio engine, the Q4 reconstruction and the eval suite are all exercised
against these numbers - inconsistent fixtures would hide real bugs.

The four issuers are shaped to cover the credit spectrum:

    NVCR  deteriorating investment grade - leverage up, margins compressing
    ARMT  strong net-cash software issuer
    KSTR  leveraged acquirer actively deleveraging
    HRBG  stressed retailer with negative free cash flow
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from creditlens.config import get_settings
from creditlens.ingest.chunker import chunk_sections
from creditlens.ingest.filing_parser import Section
from creditlens.ingest.pipeline import (
    IngestReport,
    persist_chunks,
    persist_facts,
    refresh_indexes,
    upsert_company,
    upsert_filing,
)
from creditlens.ingest.xbrl_normalizer import NormalizedFact
from creditlens.observability import get_logger

log = get_logger(__name__)

SYNTHETIC_SOURCE = "synthetic-fixture"
N_QUARTERS = 14  # 2023Q1 .. 2026Q2
START_YEAR = 2023


def ramp(start: float, end: float, n: int = N_QUARTERS) -> list[float]:
    if n == 1:
        return [end]
    step = (end - start) / (n - 1)
    return [start + step * i for i in range(n)]


def seasonal(base: list[float], factors: Sequence[float]) -> list[float]:
    return [v * factors[i % 4] for i, v in enumerate(base)]


@dataclass
class IssuerSpec:
    ticker: str
    name: str
    cik: str
    industry: str
    sic: str
    profile: str
    revenue: list[float]
    gross_margin: list[float]
    sga_pct: list[float]
    rd_pct: list[float]
    da_pct: list[float]
    total_debt: list[float]
    short_term_share: float
    debt_rate: list[float]
    capex_pct: list[float]
    wc_drag_pct: list[float]
    opening_cash: float
    opening_sti: float
    opening_equity: float
    opening_retained: float
    goodwill: float
    intangibles: float
    ppe_pct_revenue: float
    dso_days: float
    dio_days: float
    dpo_days: float
    dividend_payout: float
    buyback_pct_ocf: float
    themes: dict[str, list[str]] = field(default_factory=dict)


def _q_end(index: int) -> dt.date:
    year = START_YEAR + index // 4
    month = [3, 6, 9, 12][index % 4]
    day = {3: 31, 6: 30, 9: 30, 12: 31}[month]
    return dt.date(year, month, day)


def _q_start(index: int) -> dt.date:
    year = START_YEAR + index // 4
    month = [1, 4, 7, 10][index % 4]
    return dt.date(year, month, 1)


def build_quarters(spec: IssuerSpec) -> list[dict[str, float]]:
    """Generate articulated quarterly statements for one issuer."""
    cash = spec.opening_cash
    equity = spec.opening_equity
    retained = spec.opening_retained
    out: list[dict[str, float]] = []

    for i in range(N_QUARTERS):
        revenue = spec.revenue[i]
        cost_of_revenue = revenue * (1 - spec.gross_margin[i])
        sga = revenue * spec.sga_pct[i]
        rd = revenue * spec.rd_pct[i]
        da = revenue * spec.da_pct[i]
        operating_income = revenue - cost_of_revenue - sga - rd - da

        debt = spec.total_debt[i]
        prior_debt = spec.total_debt[i - 1] if i > 0 else spec.total_debt[0]
        interest_expense = debt * spec.debt_rate[i] / 4.0

        pretax = operating_income - interest_expense
        tax = max(pretax, 0.0) * 0.21
        net_income = pretax - tax

        working_capital_drag = revenue * spec.wc_drag_pct[i]
        operating_cash_flow = net_income + da - working_capital_drag
        capex = revenue * spec.capex_pct[i]
        free_cash_flow = operating_cash_flow - capex

        dividends = max(net_income, 0.0) * spec.dividend_payout
        buybacks = max(operating_cash_flow, 0.0) * spec.buyback_pct_ocf
        debt_change = debt - prior_debt

        cash = cash + free_cash_flow - dividends - buybacks + debt_change
        retained = retained + net_income - dividends
        equity = equity + net_income - dividends - buybacks

        receivables = revenue * spec.dso_days / 91.25
        inventory = cost_of_revenue * spec.dio_days / 91.25
        payables = cost_of_revenue * spec.dpo_days / 91.25
        ppe = revenue * spec.ppe_pct_revenue
        short_term_debt = debt * spec.short_term_share
        long_term_debt = debt - short_term_debt
        accrued = revenue * 0.16

        sti = spec.opening_sti
        current_assets = cash + sti + receivables + inventory + revenue * 0.05
        current_liabilities = payables + short_term_debt + accrued
        other_long_term_liabilities = revenue * 0.22
        total_liabilities = current_liabilities + long_term_debt + other_long_term_liabilities
        total_assets = total_liabilities + equity
        # single explicit plug so the balance sheet closes exactly
        other_assets = total_assets - (current_assets + ppe + spec.goodwill + spec.intangibles)

        out.append({
            "revenue": revenue,
            "cost_of_revenue": cost_of_revenue,
            "gross_profit": revenue - cost_of_revenue,
            "sga_expense": sga,
            "rd_expense": rd,
            "depreciation_amortization": da,
            "operating_income": operating_income,
            "interest_expense": interest_expense,
            "pretax_income": pretax,
            "income_tax_expense": tax,
            "net_income": net_income,
            "operating_cash_flow": operating_cash_flow,
            "capex": capex,
            "dividends_paid": dividends,
            "share_repurchase": buybacks,
            "cash_and_equivalents": cash,
            "short_term_investments": sti,
            "accounts_receivable": receivables,
            "inventory": inventory,
            "current_assets": current_assets,
            "total_assets": total_assets,
            "goodwill": spec.goodwill,
            "intangible_assets": spec.intangibles,
            "accounts_payable": payables,
            "current_liabilities": current_liabilities,
            "short_term_debt": short_term_debt,
            "long_term_debt": long_term_debt,
            "total_liabilities": total_liabilities,
            "total_equity": equity,
            "retained_earnings": retained,
            "_other_assets": other_assets,
            "_free_cash_flow": free_cash_flow,
        })
    return out


FLOW_CONCEPTS = (
    "revenue", "cost_of_revenue", "gross_profit", "sga_expense", "rd_expense",
    "depreciation_amortization", "operating_income", "interest_expense",
    "pretax_income", "income_tax_expense", "net_income", "operating_cash_flow",
    "capex", "dividends_paid", "share_repurchase",
)
STOCK_CONCEPTS = (
    "cash_and_equivalents", "short_term_investments", "accounts_receivable",
    "inventory", "current_assets", "total_assets", "goodwill", "intangible_assets",
    "accounts_payable", "current_liabilities", "short_term_debt", "long_term_debt",
    "total_liabilities", "total_equity", "retained_earnings",
)

STATEMENT_OF = {
    **dict.fromkeys(("revenue", "cost_of_revenue", "gross_profit", "sga_expense", "rd_expense", "operating_income", "interest_expense", "pretax_income", "income_tax_expense", "net_income"), "income"),
    **dict.fromkeys(STOCK_CONCEPTS, "balance"),
    **dict.fromkeys(("operating_cash_flow", "capex", "dividends_paid", "share_repurchase", "depreciation_amortization"), "cashflow"),
}


def facts_for(spec: IssuerSpec, quarters: list[dict[str, float]]) -> list[NormalizedFact]:
    """Emit quarterly facts plus fiscal-year aggregates."""
    facts: list[NormalizedFact] = []

    def emit(concept: str, value: float, index: int, period: str, year: int) -> None:
        is_flow = concept in FLOW_CONCEPTS
        facts.append(NormalizedFact(
            concept=concept,
            raw_concept=f"fixture:{concept}",
            statement=STATEMENT_OF.get(concept, "other"),
            value=round(value, 2),
            unit="USD",
            period_type="duration" if is_flow else "instant",
            period_start=_q_start(index) if is_flow else None,
            period_end=_q_end(index),
            fiscal_year=year,
            fiscal_period=period,
            accession=_accession(spec, year, period),
            filed=_q_end(index) + dt.timedelta(days=35),
            source=SYNTHETIC_SOURCE,
            is_synthetic=True,
        ))

    for i, quarter in enumerate(quarters):
        year = START_YEAR + i // 4
        period = f"Q{i % 4 + 1}"
        for concept in FLOW_CONCEPTS + STOCK_CONCEPTS:
            emit(concept, quarter[concept], i, period, year)

    # fiscal-year rollups for complete years only
    for year_offset in range(N_QUARTERS // 4):
        indices = list(range(year_offset * 4, year_offset * 4 + 4))
        if len(indices) < 4 or indices[-1] >= len(quarters):
            continue
        year = START_YEAR + year_offset
        last = indices[-1]
        for concept in FLOW_CONCEPTS:
            emit(concept, sum(quarters[i][concept] for i in indices), last, "FY", year)
        for concept in STOCK_CONCEPTS:
            emit(concept, quarters[last][concept], last, "FY", year)
    return facts


def _accession(spec: IssuerSpec, year: int, period: str) -> str:
    tail = {"FY": "001", "Q1": "011", "Q2": "012", "Q3": "013", "Q4": "014"}[period]
    return f"{spec.cik}-{str(year)[-2:]}-{tail}"


def _fmt_usd(value: float) -> str:
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.2f} billion"
    return f"${value:,.0f} million"


def _metrics_for(quarters: list[dict[str, float]], index: int) -> dict[str, Any]:
    """Trailing-twelve-month headline metrics used inside the narrative text."""
    window = quarters[max(0, index - 3): index + 1]
    latest = quarters[index]
    ttm_revenue = sum(q["revenue"] for q in window)
    ttm_ebitda = sum(q["operating_income"] + q["depreciation_amortization"] for q in window)
    ttm_interest = sum(q["interest_expense"] for q in window)
    ttm_fcf = sum(q["_free_cash_flow"] for q in window)
    debt = latest["short_term_debt"] + latest["long_term_debt"]
    liquid = latest["cash_and_equivalents"] + latest["short_term_investments"]
    return {
        "revenue": ttm_revenue,
        "ebitda": ttm_ebitda,
        "ebitda_margin": ttm_ebitda / ttm_revenue * 100 if ttm_revenue else 0.0,
        "operating_margin": sum(q["operating_income"] for q in window) / ttm_revenue * 100,
        "debt": debt,
        "net_debt": debt - liquid,
        "cash": liquid,
        "leverage": debt / ttm_ebitda if ttm_ebitda > 0 else float("nan"),
        "net_leverage": (debt - liquid) / ttm_ebitda if ttm_ebitda > 0 else float("nan"),
        "coverage": ttm_ebitda / ttm_interest if ttm_interest else float("nan"),
        "fcf": ttm_fcf,
        "current_ratio": latest["current_assets"] / latest["current_liabilities"],
    }


# ---------------------------------------------------------------------------
# narrative
# ---------------------------------------------------------------------------
def _risk_section(spec: IssuerSpec, metrics: dict[str, Any]) -> str:
    body = ["Item 1A. Risk Factors", ""]
    body.append(spec.themes["risk_intro"][0].format(**_fmt_ctx(spec, metrics)))
    body.append("")
    for paragraph in spec.themes["risks"]:
        body.append(paragraph.format(**_fmt_ctx(spec, metrics)))
        body.append("")
    return "\n".join(body).strip()


def _mdna_section(spec: IssuerSpec, metrics: dict[str, Any], period_label: str, item: str) -> str:
    ctx = _fmt_ctx(spec, metrics)
    ctx["period"] = period_label
    heading = (
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"
        if item == "7"
        else "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )
    body = [heading, "", "Results of Operations", ""]
    for paragraph in spec.themes["results"]:
        body.append(paragraph.format(**ctx))
        body.append("")
    body += ["Liquidity and Capital Resources", ""]
    for paragraph in spec.themes["liquidity"]:
        body.append(paragraph.format(**ctx))
        body.append("")
    return "\n".join(body).strip()


def _business_section(spec: IssuerSpec, metrics: dict[str, Any]) -> str:
    ctx = _fmt_ctx(spec, metrics)
    body = ["Item 1. Business", ""]
    for paragraph in spec.themes["business"]:
        body.append(paragraph.format(**ctx))
        body.append("")
    return "\n".join(body).strip()


def _market_risk_section(spec: IssuerSpec, metrics: dict[str, Any]) -> str:
    ctx = _fmt_ctx(spec, metrics)
    body = ["Item 7A. Quantitative and Qualitative Disclosures About Market Risk", ""]
    for paragraph in spec.themes["market_risk"]:
        body.append(paragraph.format(**ctx))
        body.append("")
    return "\n".join(body).strip()


def _fmt_ctx(spec: IssuerSpec, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": spec.name,
        "ticker": spec.ticker,
        "revenue": _fmt_usd(metrics["revenue"]),
        "ebitda": _fmt_usd(metrics["ebitda"]),
        "ebitda_margin": f"{metrics['ebitda_margin']:.1f}%",
        "operating_margin": f"{metrics['operating_margin']:.1f}%",
        "debt": _fmt_usd(metrics["debt"]),
        "net_debt": _fmt_usd(metrics["net_debt"]),
        "cash": _fmt_usd(metrics["cash"]),
        "leverage": f"{metrics['leverage']:.1f}x",
        "net_leverage": f"{metrics['net_leverage']:.1f}x",
        "coverage": f"{metrics['coverage']:.1f}x",
        "fcf": _fmt_usd(metrics["fcf"]),
        "current_ratio": f"{metrics['current_ratio']:.2f}x",
        "period": "",
    }


def build_filings(
    spec: IssuerSpec, quarters: list[dict[str, float]]
) -> list[dict[str, Any]]:
    """Three annual reports plus the four most recent quarterly reports."""
    filings: list[dict[str, Any]] = []

    for year_offset in range(N_QUARTERS // 4):
        index = year_offset * 4 + 3
        if index >= len(quarters):
            continue
        year = START_YEAR + year_offset
        metrics = _metrics_for(quarters, index)
        sections = [
            Section("1", "Business", _business_section(spec, metrics), 0, 0),
            Section("1A", "Risk Factors", _risk_section(spec, metrics), 0, 0),
            Section("7", "Management's Discussion and Analysis",
                    _mdna_section(spec, metrics, f"fiscal {year}", "7"), 0, 0),
            Section("7A", "Quantitative and Qualitative Disclosures About Market Risk",
                    _market_risk_section(spec, metrics), 0, 0),
        ]
        filings.append({
            "accession": _accession(spec, year, "FY"),
            "form_type": "10-K",
            "period_end": _q_end(index),
            "filing_date": _q_end(index) + dt.timedelta(days=45),
            "fiscal_year": year,
            "fiscal_period": "FY",
            "sections": sections,
        })

    for index in range(len(quarters) - 4, len(quarters)):
        if index < 0:
            continue
        year = START_YEAR + index // 4
        period = f"Q{index % 4 + 1}"
        if period == "Q4":
            continue  # Q4 is covered by the annual report, as in real filers
        metrics = _metrics_for(quarters, index)
        sections = [
            Section("2", "Management's Discussion and Analysis",
                    _mdna_section(spec, metrics, f"{period} {year}", "2"), 0, 0),
            Section("1A", "Risk Factors", _risk_section(spec, metrics), 0, 0),
        ]
        filings.append({
            "accession": _accession(spec, year, period),
            "form_type": "10-Q",
            "period_end": _q_end(index),
            "filing_date": _q_end(index) + dt.timedelta(days=35),
            "fiscal_year": year,
            "fiscal_period": period,
            "sections": sections,
        })
    return filings


def load_fixtures(session: Session, *, tickers: Sequence[str] | None = None) -> list[IngestReport]:
    from creditlens.ingest.fixture_data import ISSUERS

    settings = get_settings()
    reports: list[IngestReport] = []
    wanted = {t.upper() for t in tickers} if tickers else None

    for spec in ISSUERS:
        if wanted and spec.ticker not in wanted:
            continue
        started = dt.datetime.now(dt.UTC)
        quarters = build_quarters(spec)
        company = upsert_company(
            session, cik=spec.cik, ticker=spec.ticker, name=spec.name,
            industry=spec.industry, sic=spec.sic, fiscal_year_end="12-31",
            source=SYNTHETIC_SOURCE, is_synthetic=True,
        )
        report = IngestReport(
            ticker=spec.ticker, cik=spec.cik, company_name=spec.name,
            source=SYNTHETIC_SOURCE,
        )

        filing_ids: dict[str, int] = {}
        for payload in build_filings(spec, quarters):
            filing = upsert_filing(
                session, company,
                accession=payload["accession"], form_type=payload["form_type"],
                filing_date=payload["filing_date"], period_end=payload["period_end"],
                fiscal_year=payload["fiscal_year"], fiscal_period=payload["fiscal_period"],
                url=None, source=SYNTHETIC_SOURCE, is_synthetic=True,
            )
            filing_ids[payload["accession"]] = filing.id
            chunks = chunk_sections(
                payload["sections"],
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            report.chunks_written += persist_chunks(session, company, filing, chunks)
            report.filings_ingested += 1

        facts = facts_for(spec, quarters)
        report.facts_written = persist_facts(
            session, company, facts, filing_by_accession=filing_ids
        )
        report.periods = sorted({f"{f.fiscal_period}{f.fiscal_year}" for f in facts})
        report.duration_s = (dt.datetime.now(dt.UTC) - started).total_seconds()
        reports.append(report)
        log.info("fixture issuer loaded", extra=report.to_dict())

    session.commit()
    refresh_indexes()
    return reports
