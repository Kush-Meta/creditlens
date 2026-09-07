"""Data lineage: where every number actually came from.

A separate surface from analysis, and a deliberately different kind of engine.
Analysis asks *what does the evidence mean*, which is a language task. Lineage
asks *where did this number come from*, which is *not* - the answer is already
recorded, exactly, in the provenance the calculation engine attaches to every
value. Inferring it with a model would be strictly worse: slower, more
expensive, and capable of being wrong about the one thing this surface exists
to be right about.

So the lineage engine is deterministic end to end. It walks the recorded
provenance graph and resolves it to source filings:

    debt_to_ebitda 4.33x
      = total_debt / EBITDA
        total_debt  129,541,000,000   reported, tag DebtLongtermAndShorttermCombinedAmount
                                      10-K 2026-06-20  -> sec.gov/...
        EBITDA       29,900,000,000   derived: sum over Q1-Q4 2026
          Q4 2026     6,133,000,000   operating_income + D&A
            operating_income          derived-q4: FY minus Q1+Q2+Q3
              ...                     reported, tag OperatingIncomeLoss -> sec.gov/...

Every leaf terminates in either a filed XBRL fact with its accession and a link
to the filing on sec.gov, or an explicit statement that the value was derived
and by what formula. Nothing in this module can produce a number that is not
already in the store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from creditlens.db.models import Company, Filing, FinancialFact
from creditlens.finance import ratios as ratio_mod
from creditlens.finance import statements
from creditlens.finance.statements import Period
from creditlens.observability import get_logger

log = get_logger(__name__)

MAX_DEPTH = 6

#: how a value came to exist, in plain language
ORIGIN_LABELS: dict[str, str] = {
    "sec-edgar-xbrl": "reported by the filer in XBRL",
    "synthetic-fixture": "synthetic demo data (fictional issuer)",
    "derived": "computed from other values in the same period",
    "derived-ttm": "summed across a trailing twelve-month window",
    "derived-q4": "reconstructed: fiscal year minus the first three quarters",
    "deterministic-calculation": "computed by the ratio engine",
    "reported": "reported by the filer",
}


def origin_label(source: str) -> str:
    if source.startswith("carried-forward"):
        return f"carried forward from an earlier balance sheet ({source.split('from-')[-1]})"
    if source.startswith("derived-from-ytd"):
        return "differenced out of a cumulative year-to-date figure"
    return ORIGIN_LABELS.get(source, source)


@dataclass
class LineageNode:
    label: str
    value: float | None = None
    unit: str = ""
    display: str = ""
    kind: str = "reported"          # reported | derived | computed | missing
    origin: str = ""                # plain-language explanation
    formula: str | None = None
    xbrl_tag: str | None = None
    accession: str | None = None
    filing: str | None = None       # e.g. "10-K FY2026 filed 2026-06-20"
    url: str | None = None
    period: str | None = None
    is_synthetic: bool = False
    note: str | None = None
    children: list[LineageNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "display": self.display,
            "kind": self.kind,
            "origin": self.origin,
            "formula": self.formula,
            "xbrl_tag": self.xbrl_tag,
            "accession": self.accession,
            "filing": self.filing,
            "url": self.url,
            "period": self.period,
            "is_synthetic": self.is_synthetic,
            "note": self.note,
            "children": [c.to_dict() for c in self.children],
        }

    def leaves(self) -> list[LineageNode]:
        if not self.children:
            return [self]
        return [leaf for child in self.children for leaf in child.leaves()]


class LineageResolver:
    """Walks recorded provenance down to filed facts."""

    def __init__(self, session: Session, ticker: str):
        self.session = session
        self.ticker = ticker.upper()
        self.company = session.scalar(
            select(Company).where(Company.ticker == self.ticker)
        )
        quarterly = statements.load_periods(session, self.ticker, freq="quarterly")
        annual = statements.load_periods(session, self.ticker, freq="annual")
        self.periods: dict[str, Period] = {p.key: p for p in [*quarterly, *annual]}
        ttm = statements.build_ttm(quarterly)
        if ttm is not None:
            self.periods[ttm.key] = ttm
            self.periods["TTM"] = ttm
        self._filings: dict[str, Filing] = {}
        self._facts: dict[tuple[str, str, int], FinancialFact] = {}

    # -- lookups ---------------------------------------------------------
    def filing_for(self, accession: str | None) -> Filing | None:
        if not accession:
            return None
        if accession not in self._filings:
            filing = self.session.scalar(
                select(Filing).where(Filing.accession == accession)
            )
            if filing is None:
                return None
            self._filings[accession] = filing
        return self._filings.get(accession)

    def fact_for(self, concept: str, period: Period) -> FinancialFact | None:
        key = (concept, period.fiscal_period, period.fiscal_year)
        if key not in self._facts:
            if self.company is None:
                return None
            fact = self.session.scalar(
                select(FinancialFact).where(
                    FinancialFact.company_id == self.company.id,
                    FinancialFact.concept == concept,
                    FinancialFact.fiscal_period == period.fiscal_period,
                    FinancialFact.fiscal_year == period.fiscal_year,
                )
            )
            if fact is None:
                return None
            self._facts[key] = fact
        return self._facts.get(key)

    # -- resolution ------------------------------------------------------
    def resolve_ratio(self, metric: str, period_spec: str | None = None) -> LineageNode:
        period = self._period(period_spec)
        if period is None:
            return LineageNode(
                label=metric, kind="missing",
                origin=f"no periods available for {self.ticker}",
            )
        result = ratio_mod.compute(metric, period)
        root = LineageNode(
            label=result.label,
            value=result.value,
            unit=result.unit,
            display=result.display(),
            kind="computed" if result.value is not None else "missing",
            origin="computed by the ratio engine, never by a language model",
            formula=result.formula,
            period=period.key,
            is_synthetic=result.is_synthetic,
            note=result.unavailable_reason or (
                "; ".join(result.warnings) if result.warnings else None
            ),
        )
        inputs = (
            result.inputs if result.inputs
            else dict.fromkeys(ratio_mod.REGISTRY[metric].inputs, None)
        )
        for concept in inputs:
            root.children.append(self.resolve_concept(concept, period))
        return root

    def resolve_concept(
        self, concept: str, period: Period, depth: int = 0
    ) -> LineageNode:
        value = period.get(concept)
        provenance = period.provenance.get(concept)

        if value is None:
            return LineageNode(
                label=concept, kind="missing", period=period.key,
                origin=(
                    f"not reported by {self.ticker} for {period.key}, and not "
                    "derivable from what is reported"
                ),
            )

        node = LineageNode(
            label=concept,
            value=value,
            unit="USD",
            display=_money(value),
            period=period.key,
            is_synthetic=bool(provenance and provenance.is_synthetic),
        )
        if provenance is None:
            node.kind, node.origin = "reported", origin_label("reported")
            return node

        node.formula = provenance.formula
        node.origin = origin_label(provenance.source)

        if depth >= MAX_DEPTH:
            node.kind = "derived" if provenance.source.startswith("derived") else "reported"
            node.note = "lineage truncated at maximum depth"
            return node

        # (a) summed across a TTM window: recurse into the same concept per period
        if provenance.source == "derived-ttm":
            node.kind = "derived"
            for period_key in provenance.derived_from or []:
                child_period = self.periods.get(period_key)
                if child_period is not None:
                    node.children.append(
                        self.resolve_concept(concept, child_period, depth + 1)
                    )
            return node

        # (b) computed from sibling concepts in the same period
        if provenance.source == "derived":
            node.kind = "derived"
            for child_concept in provenance.derived_from or []:
                node.children.append(
                    self.resolve_concept(child_concept, period, depth + 1)
                )
            return node

        # (c) a stored fact: terminate at the filing that reported it
        fact = self.fact_for(concept, period)
        if fact is not None:
            node.xbrl_tag = fact.raw_concept
            node.accession = fact.accession
            node.origin = origin_label(fact.source)
            node.kind = "derived" if fact.source.startswith("derived") else "reported"
            node.is_synthetic = fact.is_synthetic
            filing = self.filing_for(fact.accession)
            if filing is not None:
                node.filing = (
                    f"{filing.form_type} {filing.fiscal_period or ''}"
                    f"{filing.fiscal_year or ''} filed {filing.filing_date}"
                ).replace("  ", " ")
                node.url = filing.url
            if fact.source == "derived-q4":
                node.formula = node.formula or "fiscal year minus Q1 + Q2 + Q3"
        else:
            # No fact row under this concept name - which is normal when the
            # value was mirrored from a differently-named tag (total_debt from
            # a filer-reported combined amount). The provenance carried on the
            # value already names the tag and the filing, so use it.
            node.kind = "reported"
            node.xbrl_tag = provenance.raw_concept
            node.accession = provenance.accession
            filing = self.filing_for(provenance.accession)
            if filing is not None:
                node.filing = (
                    f"{filing.form_type} {filing.fiscal_period or ''}"
                    f"{filing.fiscal_year or ''} filed {filing.filing_date}"
                ).replace("  ", " ")
                node.url = filing.url
            elif not provenance.accession:
                node.note = "no stored fact row matched this concept and period"
        return node

    def _period(self, spec: str | None) -> Period | None:
        ordered = sorted(
            {p.key: p for p in self.periods.values()}.values(), key=Period.sort_key
        )
        if not ordered:
            return None
        return statements.select_period(ordered, spec) or (
            self.periods.get("TTM") or ordered[-1]
        )


def _money(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"${value / 1e6:,.1f}M"
    return f"${value:,.0f}"


# ---------------------------------------------------------------------------
# question answering
# ---------------------------------------------------------------------------
@dataclass
class ProvenanceAnswer:
    question: str
    intent: str
    summary: str
    detail: list[str] = field(default_factory=list)
    lineage: dict[str, Any] | None = None
    table: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    engine: str = "deterministic-lineage"
    contains_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "summary": self.summary,
            "detail": self.detail,
            "lineage": self.lineage,
            "table": self.table,
            "citations": self.citations,
            "followups": self.followups,
            "engine": self.engine,
            "contains_synthetic": self.contains_synthetic,
        }


#: Analyst phrasing -> ratio name. Reuses the verifier's vocabulary idea: people
#: do not type `net_debt_to_ebitda`.
_METRIC_PHRASES: dict[str, str] = {
    "net leverage": "net_debt_to_ebitda", "net debt to ebitda": "net_debt_to_ebitda",
    "gross leverage": "debt_to_ebitda", "leverage": "debt_to_ebitda",
    "debt to ebitda": "debt_to_ebitda", "debt/ebitda": "debt_to_ebitda",
    "interest coverage": "ebitda_interest_coverage", "coverage": "ebitda_interest_coverage",
    "current ratio": "current_ratio", "quick ratio": "quick_ratio",
    "cash ratio": "cash_ratio", "cash to debt": "cash_to_debt",
    "operating margin": "operating_margin", "gross margin": "gross_margin",
    "ebitda margin": "ebitda_margin", "net margin": "net_margin",
    "free cash flow margin": "fcf_margin", "fcf margin": "fcf_margin",
    "return on assets": "return_on_assets", "return on equity": "return_on_equity",
    "debt to equity": "debt_to_equity", "debt to capital": "debt_to_capital",
    "asset turnover": "asset_turnover",
}

_CONCEPT_PHRASES: dict[str, str] = {
    "revenue": "revenue", "sales": "revenue", "total debt": "total_debt",
    "net debt": "net_debt", "ebitda": "ebitda", "operating income": "operating_income",
    "net income": "net_income", "free cash flow": "free_cash_flow",
    "operating cash flow": "operating_cash_flow", "capex": "capex",
    "capital expenditure": "capex", "cash": "cash_and_equivalents",
    "total assets": "total_assets", "total equity": "total_equity",
    "interest expense": "interest_expense", "inventory": "inventory",
}

_PERIOD_RE = re.compile(r"\b(TTM|FY\s?\d{4}|Q[1-4][\s\-]?\d{4})\b", re.I)


def _find_ticker(question: str, known: list[str], names: dict[str, str]) -> str | None:
    """Resolve the issuer a question is about.

    Order matters. One- and two-letter tickers (F, T, M, O) match inside
    ordinary words and contractions - a plain word-boundary search finds AT&T
    in "why can't you..." - so they are tried last and only when they appear as
    a standalone capitalised token in the original text.
    """
    upper = question.upper()

    for ticker in sorted((t for t in known if len(t) >= 3), key=len, reverse=True):
        if re.search(rf"\b{re.escape(ticker)}\b", upper):
            return ticker

    for ticker, name in names.items():
        head = re.sub(r"[^A-Z ]", "", name.upper()).split()
        if head and len(head[0]) > 3 and head[0][:5] in upper:
            return ticker

    for ticker in sorted((t for t in known if len(t) < 3), key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z'’]){re.escape(ticker)}(?![A-Za-z'’])", question):
            return ticker
    return None


def _normalize(question: str) -> str:
    """Lowercase and flatten separators, so 'debt-to-EBITDA' matches 'debt to ebitda'."""
    return re.sub(r"[\-_/]+", " ", question.lower())


def _find_metric(question: str) -> str | None:
    lowered = _normalize(question)
    for phrase, metric in sorted(_METRIC_PHRASES.items(), key=lambda kv: -len(kv[0])):
        if phrase in lowered:
            return metric
    for name in ratio_mod.REGISTRY:
        if name.replace("_", " ") in lowered or name in lowered:
            return name
    return None


def _find_concept(question: str) -> str | None:
    lowered = _normalize(question)
    for phrase, concept in sorted(_CONCEPT_PHRASES.items(), key=lambda kv: -len(kv[0])):
        if phrase in lowered:
            return concept
    return None


def _find_period(question: str) -> str | None:
    match = _PERIOD_RE.search(question)
    return match.group(1).replace(" ", "").upper() if match else None


PIPELINE_TOPICS: dict[str, tuple[str, list[str]]] = {
    "source": (
        "Everything comes from SEC EDGAR: XBRL company facts for numbers, and "
        "10-K/10-Q documents for narrative.",
        [
            "Numbers come from the XBRL `companyfacts` API, normalised into canonical "
            "concepts so that filer-specific tags (Revenues vs "
            "RevenueFromContractWithCustomerExcludingAssessedTax) collapse to one name.",
            "Narrative comes from the filing documents themselves, split into SEC item "
            "sections (1A Risk Factors, 7 MD&A, 8 Financial Statements) and chunked.",
            "Every stored fact keeps the original XBRL tag in `raw_concept`, the "
            "accession number of the filing that reported it, and a source label.",
            "Demo issuers NVCR, ARMT, KSTR and HRBG are fictional with synthetic "
            "financials, and every value from them is flagged `is_synthetic`.",
        ],
    ),
    "calculation": (
        "Ratios are computed by deterministic code, never by a language model.",
        [
            "Each ratio is a declared object with its formula, required inputs and "
            "guards, so every result carries the exact inputs it consumed.",
            "A ratio with a meaningless denominator returns no value plus a reason, "
            "rather than a number that looks authoritative.",
            "Quarterly flows are annualised where required, and stock/flow confusion "
            "is impossible: a balance sheet can never be summed across quarters.",
        ],
    ),
    "verification": (
        "Every figure in an answer is checked back against an evidence ledger after "
        "the answer is written.",
        [
            "Tool output is registered in an evidence ledger before it re-enters the "
            "conversation.",
            "Each number in the narrative is then extracted and classified: verified "
            "(matches a computed value), cited (quoted from a retrieved passage), "
            "contradicted (near a computed value but outside tolerance), or "
            "unsupported (traceable to nothing).",
            "Confidence is capped by that result - a model may lower its stated "
            "confidence but never raise it above what the evidence supports.",
        ],
    ),
    "retrieval": (
        "Filing text is retrieved by a hybrid of BM25 and dense vectors, fused by "
        "reciprocal rank fusion.",
        [
            "Metadata filters narrow by issuer, form, fiscal year and SEC item before "
            "anything is scored.",
            "Fusion is by rank rather than score, so unbounded BM25 scores and bounded "
            "cosine never need calibrating.",
            "A per-issuer quota guarantees comparison questions get evidence from both "
            "sides rather than eight passages about one company.",
        ],
    ),
    "derived": (
        "Some values are derived rather than reported, and each says so.",
        [
            "`derived-q4`: no company files a 10-Q for the fourth quarter, so Q4 flows "
            "are reconstructed as the fiscal year minus the first three quarters.",
            "`derived-from-ytd`: 10-Q cash-flow statements are cumulative, so discrete "
            "quarters are differenced out of the year-to-date figures.",
            "`derived-ttm`: trailing-twelve-month flows are summed over four quarters, "
            "with balance-sheet items taken from the most recent quarter.",
            "`carried-forward`: a balance reported at an earlier date, reused with its "
            "staleness recorded.",
        ],
    ),
}


def answer(session: Session, question: str) -> ProvenanceAnswer:
    """Answer a question about where the data comes from.

    Deterministic throughout: intent is matched by pattern, and every fact in the
    reply is read from the store rather than generated.
    """
    companies = list(session.scalars(select(Company).order_by(Company.ticker)))
    known = [c.ticker for c in companies]
    names = {c.ticker: c.name for c in companies}
    lowered = question.lower()

    ticker = _find_ticker(question, known, names)
    metric = _find_metric(question)
    # a metric phrase wins over a concept nested inside it: "debt-to-EBITDA"
    # contains "ebitda", but the question is about the ratio
    concept = None if metric else _find_concept(question)
    period = _find_period(question)

    # explicit "how does the system work" questions
    if not ticker or re.search(r"how (do|does) (you|the system|creditlens|this)", lowered):
        topic = _pipeline_topic(lowered)
        if topic and not (ticker and (metric or concept)):
            return _pipeline_answer(question, topic)

    if ticker is None:
        return _corpus_answer(session, question, companies)

    if metric:
        return _metric_answer(session, question, ticker, metric, period)
    if concept:
        return _concept_answer(session, question, ticker, concept, period)
    if re.search(r"filing|10-k|10-q|document|report", lowered):
        return _filings_answer(session, question, ticker)
    return _coverage_answer(session, question, ticker)


def _pipeline_topic(lowered: str) -> str | None:
    if re.search(r"verif|check|hallucinat|ground|trust|made up|fabricat", lowered):
        return "verification"
    if re.search(r"retriev|search|passage|chunk|citation", lowered):
        return "retrieval"
    if re.search(r"derived|reconstruct|q4|ytd|trailing|ttm", lowered):
        return "derived"
    if re.search(r"calculat|comput|ratio|formula", lowered):
        return "calculation"
    if re.search(r"where.*(data|number|come from)|source|edgar|xbrl", lowered):
        return "source"
    return None


def _pipeline_answer(question: str, topic: str) -> ProvenanceAnswer:
    summary, detail = PIPELINE_TOPICS[topic]
    return ProvenanceAnswer(
        question=question, intent=f"pipeline:{topic}", summary=summary, detail=detail,
        followups=[
            "Where does Oracle's debt-to-EBITDA come from?",
            "What data do you have for Ford?",
            "How are Q4 figures derived?",
        ],
    )


def _metric_answer(
    session: Session, question: str, ticker: str, metric: str, period: str | None
) -> ProvenanceAnswer:
    resolver = LineageResolver(session, ticker)
    root = resolver.resolve_ratio(metric, period)
    leaves = [n for n in root.leaves() if n.kind != "missing"]
    filings = {
        (n.accession, n.filing, n.url) for n in root.leaves() if n.accession
    }

    if root.value is None:
        summary = (
            f"{ticker} {root.label} cannot be computed for {root.period}. "
            f"{root.note or 'Required inputs are missing.'}"
        )
    else:
        summary = (
            f"{ticker} {root.label} is {root.display} for {root.period}, computed as "
            f"{root.formula}. It traces to {len(leaves)} underlying value(s) across "
            f"{len(filings)} filing(s)."
        )

    detail: list[str] = []
    if root.value is None:
        # "missing inputs: total_debt" is accurate but unhelpful. Say which
        # concepts are absent for this issuer and why that happens.
        missing = re.findall(r"missing inputs: (.+)", root.note or "")
        absent = [c.strip() for c in missing[0].split(",")] if missing else []
        if absent:
            detail.append(
                f"{ticker} does not report {', '.join(absent)} in its XBRL facts. "
                "Filers sometimes tag these only at segment level through custom "
                "taxonomy extensions, which the companyfacts API does not expose."
            )
        detail.append(
            "The value is reported as unavailable rather than estimated. Guessing a "
            "leverage ratio would be worse than not having one."
        )
    else:
        detail.append(
            "This figure was computed by the ratio engine. No language model produced "
            "or adjusted it."
        )
    derived = [n for n in root.leaves() if n.kind == "derived"]
    if derived:
        kinds = {n.origin for n in derived}
        detail.append(
            "Some inputs are derived rather than directly reported: "
            + "; ".join(sorted(kinds)) + "."
        )
    if root.is_synthetic:
        detail.append(
            "This issuer is a fictional demo company; the figures are synthetic and "
            "are not filed results."
        )

    return ProvenanceAnswer(
        question=question, intent="metric_lineage", summary=summary, detail=detail,
        lineage=root.to_dict(),
        citations=[
            {"accession": accession, "filing": filing, "url": url}
            for accession, filing, url in sorted(filings, key=lambda x: str(x[1]))
        ],
        followups=[
            f"What data do you have for {ticker}?",
            f"Where does {ticker}'s total debt come from?",
            "How are Q4 figures derived?",
        ],
        contains_synthetic=root.is_synthetic,
    )


def _concept_answer(
    session: Session, question: str, ticker: str, concept: str, period: str | None
) -> ProvenanceAnswer:
    resolver = LineageResolver(session, ticker)
    target = resolver._period(period)
    if target is None:
        return ProvenanceAnswer(
            question=question, intent="concept_lineage",
            summary=f"No periods are available for {ticker}.",
        )
    node = resolver.resolve_concept(concept, target)
    if node.value is None:
        return ProvenanceAnswer(
            question=question, intent="concept_lineage",
            summary=f"{ticker} does not report {concept} for {target.key}. {node.origin}",
            followups=[f"What data do you have for {ticker}?"],
        )
    origin = node.origin
    if node.xbrl_tag:
        origin += f", tagged `{node.xbrl_tag}`"
    summary = f"{ticker} {concept} for {target.key} is {node.display} — {origin}."
    citations = [
        {"accession": leaf.accession, "filing": leaf.filing, "url": leaf.url}
        for leaf in node.leaves() if leaf.accession
    ]
    return ProvenanceAnswer(
        question=question, intent="concept_lineage", summary=summary,
        detail=[node.formula] if node.formula else [],
        lineage=node.to_dict(),
        citations=_dedupe(citations),
        followups=[
            f"Where does {ticker}'s leverage come from?",
            f"What filings do you have for {ticker}?",
        ],
        contains_synthetic=node.is_synthetic,
    )


def _coverage_answer(session: Session, question: str, ticker: str) -> ProvenanceAnswer:
    from creditlens.agent.evidence import EvidenceLedger
    from creditlens.agent.tools import ToolContext, ToolError, tool_data_coverage
    from creditlens.observability import NullTrace

    context = ToolContext(session=session, ledger=EvidenceLedger(), trace=NullTrace())
    try:
        coverage = tool_data_coverage(context, ticker=ticker)
    except ToolError as exc:
        return ProvenanceAnswer(
            question=question, intent="coverage", summary=str(exc)
        )

    missing = coverage["concepts_missing"]
    summary = (
        f"{ticker}: {len(coverage['quarterly_periods'])} quarterly and "
        f"{len(coverage['annual_periods'])} annual periods, "
        f"{coverage['ratios_available']} of "
        f"{coverage['ratios_available'] + len(coverage['ratios_blocked'])} ratios "
        f"computable at {coverage['reference_period']}."
    )
    detail = [f"Data source: {coverage['data_source']}."]
    if missing:
        detail.append(
            f"Not reported in this issuer's XBRL facts: {', '.join(missing)}. "
            "Filers sometimes tag these only at segment level through custom "
            "taxonomy extensions, which the companyfacts API does not expose."
        )
        detail.append(
            f"That blocks {len(coverage['ratios_blocked'])} ratio(s). They are "
            "reported as unavailable rather than estimated."
        )
    return ProvenanceAnswer(
        question=question, intent="coverage", summary=summary, detail=detail,
        table=[
            {"ratio": entry["ratio"], "reason": entry["reason"]}
            for entry in coverage["ratios_blocked"][:12]
        ],
        followups=[
            f"Where does {ticker}'s revenue come from?",
            f"What filings do you have for {ticker}?",
        ],
        contains_synthetic=coverage["is_synthetic"],
    )


def _filings_answer(session: Session, question: str, ticker: str) -> ProvenanceAnswer:
    company = session.scalar(select(Company).where(Company.ticker == ticker))
    if company is None:
        return ProvenanceAnswer(
            question=question, intent="filings", summary=f"{ticker} is not in the corpus."
        )
    filings = list(session.scalars(
        select(Filing).where(Filing.company_id == company.id)
        .order_by(Filing.filing_date.desc())
    ))
    return ProvenanceAnswer(
        question=question, intent="filings",
        summary=(
            f"{len(filings)} filings ingested for {ticker} ({company.name}), "
            f"from {filings[-1].filing_date} to {filings[0].filing_date}."
            if filings else f"No filings ingested for {ticker}."
        ),
        table=[
            {
                "form": f.form_type,
                "period": f"{f.fiscal_period or ''}{f.fiscal_year or ''}",
                "period_end": str(f.period_end),
                "filed": str(f.filing_date),
                "accession": f.accession,
                "url": f.url,
            }
            for f in filings
        ],
        followups=[f"What data do you have for {ticker}?"],
        contains_synthetic=company.is_synthetic,
    )


def _corpus_answer(
    session: Session, question: str, companies: list[Company]
) -> ProvenanceAnswer:
    from sqlalchemy import func

    from creditlens.db.models import Chunk

    chunks = session.scalar(select(func.count(Chunk.id))) or 0
    facts = session.scalar(select(func.count(FinancialFact.id))) or 0
    filings = session.scalar(select(func.count(Filing.id))) or 0
    real = [c for c in companies if not c.is_synthetic]
    synthetic = [c for c in companies if c.is_synthetic]

    return ProvenanceAnswer(
        question=question, intent="corpus",
        summary=(
            f"{len(companies)} issuers ({len(real)} real from SEC EDGAR, "
            f"{len(synthetic)} fictional demo companies), {filings} filings, "
            f"{chunks:,} text chunks and {facts:,} financial facts."
        ),
        detail=[
            "Real issuers are ingested from SEC EDGAR: XBRL company facts for the "
            "numbers, 10-K and 10-Q documents for the narrative.",
            (
                "Fictional demo issuers (" + ", ".join(c.ticker for c in synthetic) +
                ") carry synthetic financials so the system can be demonstrated and "
                "tested offline. Every value from them is flagged."
            ) if synthetic else "",
        ],
        table=[
            {
                "ticker": c.ticker, "name": c.name, "sector": c.sector or "-",
                "profile": c.profile or "-", "source": c.source,
                "synthetic": c.is_synthetic,
            }
            for c in companies
        ],
        followups=[
            "Where does Oracle's debt-to-EBITDA come from?",
            "How do you verify the numbers?",
            "What data do you have for Ford?",
        ],
        contains_synthetic=bool(synthetic),
    )


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("accession"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
