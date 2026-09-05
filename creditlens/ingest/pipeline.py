"""Ingestion orchestration and persistence primitives.

Two entry points share the same persistence layer:

* `ingest_ticker()` - the real path: EDGAR XBRL facts + filing documents.
* `creditlens.ingest.fixtures.load_fixtures()` - the offline demo corpus.

Both go through `persist_*` so provenance, dedup, chunking and embedding
behave identically. Everything is idempotent: re-running an ingest updates in
place rather than duplicating, which matters because ingestion is the step
most likely to be re-run after a failure.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from creditlens.config import get_settings
from creditlens.db.models import Chunk, Company, Embedding, Filing, FinancialFact
from creditlens.ingest.chunker import TextChunk, chunk_sections
from creditlens.ingest.edgar_client import EdgarClient, FilingRef
from creditlens.ingest.filing_parser import Section, parse_filing_html
from creditlens.ingest.xbrl_normalizer import NormalizedFact, normalize_company_facts
from creditlens.observability import METRICS, Trace, get_logger
from creditlens.retrieval import bm25 as bm25_mod
from creditlens.retrieval import corpus as corpus_mod
from creditlens.retrieval.embeddings import get_embedder, pack

log = get_logger(__name__)


@dataclass
class IngestReport:
    ticker: str
    cik: str | None = None
    company_name: str = ""
    filings_ingested: int = 0
    chunks_written: int = 0
    facts_written: int = 0
    periods: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    source: str = "sec-edgar"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "company": self.company_name,
            "source": self.source,
            "filings_ingested": self.filings_ingested,
            "chunks_written": self.chunks_written,
            "facts_written": self.facts_written,
            "periods": self.periods,
            "warnings": self.warnings,
            "duration_s": round(self.duration_s, 2),
        }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def upsert_company(
    session: Session,
    *,
    cik: str,
    ticker: str,
    name: str,
    industry: str | None = None,
    sic: str | None = None,
    sector: str | None = None,
    profile: str | None = None,
    fiscal_year_end: str | None = None,
    source: str = "sec-edgar",
    is_synthetic: bool = False,
) -> Company:
    company = session.scalar(select(Company).where(Company.cik == cik))
    if company is None:
        company = session.scalar(select(Company).where(Company.ticker == ticker.upper()))
    if company is None:
        company = Company(cik=cik, ticker=ticker.upper(), name=name)
        session.add(company)
    company.ticker = ticker.upper()
    company.name = name
    company.cik = cik
    company.industry = industry or company.industry
    company.sic = sic or company.sic
    company.sector = sector or company.sector
    company.profile = profile or company.profile
    company.fiscal_year_end = fiscal_year_end or company.fiscal_year_end
    company.source = source
    company.is_synthetic = is_synthetic
    session.flush()
    return company


def upsert_filing(
    session: Session,
    company: Company,
    *,
    accession: str,
    form_type: str,
    filing_date: dt.date,
    period_end: dt.date | None,
    fiscal_year: int | None,
    fiscal_period: str | None,
    url: str | None,
    source: str = "sec-edgar",
    is_synthetic: bool = False,
) -> Filing:
    filing = session.scalar(select(Filing).where(Filing.accession == accession))
    if filing is None:
        filing = Filing(company_id=company.id, accession=accession, form_type=form_type,
                        filing_date=filing_date)
        session.add(filing)
    filing.company_id = company.id
    filing.form_type = form_type
    filing.filing_date = filing_date
    filing.period_end = period_end
    filing.fiscal_year = fiscal_year
    filing.fiscal_period = fiscal_period
    filing.url = url
    filing.source = source
    filing.is_synthetic = is_synthetic
    session.flush()
    return filing


def persist_chunks(
    session: Session, company: Company, filing: Filing, chunks: Sequence[TextChunk]
) -> int:
    """Replace a filing's chunks and their embeddings atomically."""
    existing = list(session.scalars(select(Chunk.id).where(Chunk.filing_id == filing.id)))
    if existing:
        session.execute(delete(Embedding).where(Embedding.chunk_id.in_(existing)))
        session.execute(delete(Chunk).where(Chunk.filing_id == filing.id))
        session.flush()

    if not chunks:
        return 0

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])
    rows: list[Chunk] = []
    for chunk in chunks:
        rows.append(Chunk(
            filing_id=filing.id,
            company_id=company.id,
            ordinal=chunk.ordinal,
            item_code=chunk.item_code,
            section_title=chunk.section_title,
            heading_path=chunk.heading_path,
            text=chunk.text,
            token_count=chunk.token_count,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            content_hash=chunk.content_hash,
        ))
    session.add_all(rows)
    session.flush()

    session.add_all([
        Embedding(chunk_id=row.id, model=embedder.name, dim=embedder.dim,
                  vector=pack(vectors[i]))
        for i, row in enumerate(rows)
    ])
    session.flush()
    METRICS.inc("creditlens_chunks_ingested_total", value=len(rows))
    return len(rows)


def persist_facts(
    session: Session,
    company: Company,
    facts: Sequence[NormalizedFact],
    *,
    filing_by_accession: dict[str, int] | None = None,
) -> int:
    """Insert or update normalized facts, keyed on the natural period key."""
    filing_by_accession = filing_by_accession or {}
    existing = {
        (f.concept, f.fiscal_year, f.fiscal_period, f.period_end): f
        for f in session.scalars(
            select(FinancialFact).where(FinancialFact.company_id == company.id)
        )
    }
    written = 0
    for fact in facts:
        key = (fact.concept, fact.fiscal_year, fact.fiscal_period, fact.period_end)
        row = existing.get(key)
        if row is None:
            row = FinancialFact(
                company_id=company.id, concept=fact.concept, fiscal_year=fact.fiscal_year,
                fiscal_period=fact.fiscal_period, period_end=fact.period_end, value=fact.value,
            )
            session.add(row)
            existing[key] = row
        row.value = fact.value
        row.raw_concept = fact.raw_concept
        row.statement = fact.statement
        row.unit = fact.unit
        row.period_type = fact.period_type
        row.period_start = fact.period_start
        row.accession = fact.accession
        row.source = fact.source
        row.is_synthetic = fact.is_synthetic
        row.filing_id = filing_by_accession.get(fact.accession or "")
        written += 1
    session.flush()
    METRICS.inc("creditlens_facts_ingested_total", value=written)
    return written


def refresh_indexes() -> None:
    """Drop cached retrieval structures so new documents are searchable."""
    corpus_mod.invalidate()
    bm25_mod.invalidate()


# ---------------------------------------------------------------------------
# EDGAR path
# ---------------------------------------------------------------------------
@dataclass
class FetchedFiling:
    ref: FilingRef
    sections: list[Section]


@dataclass
class FetchedIssuer:
    """Everything one issuer needs, gathered without touching the database."""

    ticker: str
    cik: str
    name: str
    facts: list[NormalizedFact]
    filings: list[FetchedFiling]
    warnings: list[str] = field(default_factory=list)
    fetch_seconds: float = 0.0


def fetch_issuer(
    client: EdgarClient,
    ticker: str,
    *,
    max_filings: int = 6,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    min_fiscal_year: int | None = None,
    trace: Trace | None = None,
) -> FetchedIssuer:
    """Network and parsing only.

    Kept free of database access so it can run on a worker thread: EDGAR
    ingestion is dominated by download latency, while SQLite tolerates exactly
    one writer. Fetch fans out, persistence stays on one thread.
    """
    started = dt.datetime.now(dt.UTC)
    trace = trace or Trace()
    warnings: list[str] = []

    with trace.span("edgar.resolve_cik", ticker=ticker):
        cik, name = client.resolve_cik(ticker)
    with trace.span("edgar.company_facts", cik=cik):
        payload = client.company_facts(cik)
    with trace.span("xbrl.normalize", ticker=ticker):
        facts = normalize_company_facts(payload, min_fiscal_year=min_fiscal_year)
    with trace.span("edgar.submissions", cik=cik):
        refs = client.recent_filings(cik, forms=forms, limit=max_filings)

    settings = get_settings()
    fetched: list[FetchedFiling] = []
    for ref in refs:
        try:
            with trace.span("edgar.document", accession=ref.accession, form=ref.form_type):
                html = client.filing_document(ref)
            sections = parse_filing_html(html)
        except Exception as exc:
            warnings.append(f"{ref.accession}: document unavailable ({exc})")
            log.warning("filing document failed",
                        extra={"ticker": ticker, "accession": ref.accession, "error": str(exc)})
            continue
        fetched.append(FetchedFiling(ref=ref, sections=sections))

    _ = settings  # settings are read during persistence; touched here for clarity
    return FetchedIssuer(
        ticker=ticker.upper(), cik=cik, name=name, facts=facts, filings=fetched,
        warnings=warnings,
        fetch_seconds=(dt.datetime.now(dt.UTC) - started).total_seconds(),
    )


def persist_issuer(
    session: Session,
    issuer: FetchedIssuer,
    *,
    sector: str | None = None,
    profile: str | None = None,
) -> IngestReport:
    """Write one fetched issuer. Single-threaded by design."""
    settings = get_settings()
    report = IngestReport(
        ticker=issuer.ticker, cik=issuer.cik, company_name=issuer.name,
        warnings=list(issuer.warnings),
    )
    company = upsert_company(
        session, cik=issuer.cik, ticker=issuer.ticker, name=issuer.name,
        sector=sector, profile=profile,
    )

    filing_ids: dict[str, int] = {}
    for item in issuer.filings:
        ref = item.ref
        period_end = _date(ref.report_date)
        fiscal_year, fiscal_period = _fiscal_from_ref(ref, period_end)
        filing = upsert_filing(
            session, company,
            accession=ref.accession, form_type=ref.form_type,
            filing_date=_date(ref.filing_date) or dt.date.today(),
            period_end=period_end, fiscal_year=fiscal_year, fiscal_period=fiscal_period,
            url=ref.document_url(settings.edgar_www_url),
        )
        filing_ids[ref.accession] = filing.id
        chunks = chunk_sections(
            item.sections,
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        report.chunks_written += persist_chunks(session, company, filing, chunks)
        report.filings_ingested += 1

    report.facts_written = persist_facts(
        session, company, issuer.facts, filing_by_accession=filing_ids
    )
    report.periods = sorted({f"{f.fiscal_period}{f.fiscal_year}" for f in issuer.facts})
    report.duration_s = issuer.fetch_seconds
    return report


def ingest_ticker(
    session: Session,
    ticker: str,
    *,
    max_filings: int = 6,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    min_fiscal_year: int | None = None,
    client: EdgarClient | None = None,
    trace: Trace | None = None,
    sector: str | None = None,
    profile: str | None = None,
) -> IngestReport:
    """Ingest one issuer end-to-end from SEC EDGAR."""
    started = dt.datetime.now(dt.UTC)
    owns_client = client is None
    client = client or EdgarClient()
    try:
        fetched = fetch_issuer(
            client, ticker, max_filings=max_filings, forms=forms,
            min_fiscal_year=min_fiscal_year, trace=trace,
        )
        report = persist_issuer(session, fetched, sector=sector, profile=profile)
        session.commit()
        refresh_indexes()
    finally:
        if owns_client:
            client.close()

    report.duration_s = (dt.datetime.now(dt.UTC) - started).total_seconds()
    log.info("ingest complete", extra=report.to_dict())
    return report


@dataclass
class UniverseReport:
    requested: int = 0
    ingested: int = 0
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    reports: list[IngestReport] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def totals(self) -> dict[str, int]:
        return {
            "filings": sum(r.filings_ingested for r in self.reports),
            "chunks": sum(r.chunks_written for r in self.reports),
            "facts": sum(r.facts_written for r in self.reports),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "ingested": self.ingested,
            "skipped": self.skipped,
            "failed": self.failed,
            "totals": self.totals,
            "duration_s": round(self.duration_s, 1),
            "issuers": [r.to_dict() for r in self.reports],
        }


def ingest_universe(
    session_factory_fn,
    issuers: Sequence[Any],
    *,
    max_filings: int = 12,
    min_fiscal_year: int | None = None,
    workers: int = 4,
    resume: bool = True,
    progress=None,
) -> UniverseReport:
    """Ingest a whole universe: fetch in parallel, persist on one thread.

    `resume` skips issuers that already hold at least `max_filings` filings, so
    an interrupted run continues instead of re-downloading. EDGAR's rate limit
    is enforced by the shared client, not by the worker count.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    started = dt.datetime.now(dt.UTC)
    report = UniverseReport(requested=len(issuers))
    client = EdgarClient()

    pending: list[Any] = []
    with session_factory_fn() as session:
        for issuer in issuers:
            ticker = getattr(issuer, "ticker", issuer)
            if resume and _already_ingested(session, ticker, max_filings):
                report.skipped.append(ticker)
                continue
            pending.append(issuer)

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    fetch_issuer, client, getattr(issuer, "ticker", issuer),
                    max_filings=max_filings, min_fiscal_year=min_fiscal_year,
                ): issuer
                for issuer in pending
            }
            done = 0
            for future in as_completed(futures):
                issuer = futures[future]
                ticker = getattr(issuer, "ticker", issuer)
                done += 1
                try:
                    fetched = future.result()
                except Exception as exc:
                    report.failed.append(f"{ticker}: {exc}")
                    log.warning("issuer fetch failed", extra={"ticker": ticker, "error": str(exc)})
                    if progress:
                        progress(done, len(pending), ticker, None)
                    continue
                with session_factory_fn() as session:
                    issuer_report = persist_issuer(
                        session, fetched,
                        sector=getattr(issuer, "sector", None),
                        profile=getattr(issuer, "profile", None),
                    )
                report.reports.append(issuer_report)
                report.ingested += 1
                if progress:
                    progress(done, len(pending), ticker, issuer_report)
    finally:
        client.close()
        refresh_indexes()

    report.duration_s = (dt.datetime.now(dt.UTC) - started).total_seconds()
    log.info("universe ingest complete", extra={
        "requested": report.requested, "ingested": report.ingested,
        "skipped": len(report.skipped), "failed": len(report.failed),
        **report.totals, "duration_s": round(report.duration_s, 1),
    })
    return report


def _already_ingested(session: Session, ticker: str, min_filings: int) -> bool:
    company = session.scalar(select(Company).where(Company.ticker == ticker.upper()))
    if company is None:
        return False
    count = session.scalar(
        select(func.count(Filing.id)).where(Filing.company_id == company.id)
    ) or 0
    return count >= min_filings


def _date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _fiscal_from_ref(ref: FilingRef, period_end: dt.date | None) -> tuple[int | None, str | None]:
    if period_end is None:
        return None, None
    if ref.form_type == "10-K":
        return period_end.year, "FY"
    return period_end.year, f"Q{(period_end.month - 1) // 3 + 1}"


def delete_company(session: Session, ticker: str) -> bool:
    company = session.scalar(select(Company).where(Company.ticker == ticker.upper()))
    if company is None:
        return False
    session.delete(company)
    session.commit()
    refresh_indexes()
    return True
