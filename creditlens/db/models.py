"""Relational schema.

Design notes
------------
* Structured financials (`FinancialFact`) and unstructured text (`Chunk`) live
  in the same database and are joined through `Filing`, so a narrative claim
  can always be traced back to the same document that produced a number.
* Everything carries `source` + `is_synthetic`. A credit opinion computed on
  demo fixtures must never be presentable as one computed on real filings.
* Vectors are stored as float32 blobs. At this corpus size an in-process numpy
  scan beats a vector database on both latency and operational surface; the
  `VectorStore` interface is the seam to swap in pgvector/Qdrant later.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    cik: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(256))
    sic: Mapped[str | None] = mapped_column(String(8), default=None)
    industry: Mapped[str | None] = mapped_column(String(128), default=None)
    sector: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    #: why the issuer is in the curated universe - a sampling label, never a rating
    profile: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    fiscal_year_end: Mapped[str | None] = mapped_column(String(8), default=None)  # MM-DD
    source: Mapped[str] = mapped_column(String(32), default="sec-edgar")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    filings: Mapped[list[Filing]] = relationship(back_populates="company", cascade="all, delete-orphan")
    facts: Mapped[list[FinancialFact]] = relationship(back_populates="company", cascade="all, delete-orphan")


class Filing(Base):
    __tablename__ = "filings"
    __table_args__ = (UniqueConstraint("accession", name="uq_filing_accession"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), index=True)
    accession: Mapped[str] = mapped_column(String(32), index=True)
    form_type: Mapped[str] = mapped_column(String(16), index=True)  # 10-K, 10-Q, 8-K
    filing_date: Mapped[dt.date] = mapped_column(Date, index=True)
    period_end: Mapped[dt.date | None] = mapped_column(Date, default=None, index=True)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, default=None)
    fiscal_period: Mapped[str | None] = mapped_column(String(4), default=None)  # FY,Q1..Q4
    url: Mapped[str | None] = mapped_column(String(512), default=None)
    source: Mapped[str] = mapped_column(String(32), default="sec-edgar")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    company: Mapped[Company] = relationship(back_populates="filings")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="filing", cascade="all, delete-orphan")

    @property
    def label(self) -> str:
        period = self.fiscal_period or ""
        year = self.fiscal_year or (self.period_end.year if self.period_end else "")
        return f"{self.form_type} {period}{year}".strip()


class Chunk(Base):
    """A retrievable passage of filing text."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunk_filing_ordinal", "filing_id", "ordinal"),
        Index("ix_chunk_item", "item_code"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id", ondelete="CASCADE"), index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    item_code: Mapped[str | None] = mapped_column(String(16), default=None)  # e.g. "1A", "7"
    section_title: Mapped[str | None] = mapped_column(String(256), default=None)
    heading_path: Mapped[str | None] = mapped_column(String(512), default=None)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(40), index=True)

    filing: Mapped[Filing] = relationship(back_populates="chunks")
    embedding: Mapped[Embedding | None] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", uselist=False
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(64))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)

    chunk: Mapped[Chunk] = relationship(back_populates="embedding")


class FinancialFact(Base):
    """One normalized financial datapoint for one company-period.

    `concept` is the canonical CreditLens concept name (see finance.taxonomy),
    `raw_concept` keeps the source XBRL tag so normalization is auditable.
    """

    __tablename__ = "financial_facts"
    __table_args__ = (
        Index("ix_fact_lookup", "company_id", "concept", "fiscal_year", "fiscal_period"),
        UniqueConstraint(
            "company_id", "concept", "fiscal_year", "fiscal_period", "period_end",
            name="uq_fact_period",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), index=True)
    filing_id: Mapped[int | None] = mapped_column(ForeignKey("filings.id", ondelete="SET NULL"), default=None)
    concept: Mapped[str] = mapped_column(String(64), index=True)
    raw_concept: Mapped[str | None] = mapped_column(String(128), default=None)
    statement: Mapped[str] = mapped_column(String(24), default="other")  # income/balance/cashflow
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16), default="USD")
    period_type: Mapped[str] = mapped_column(String(8), default="duration")  # duration|instant
    period_start: Mapped[dt.date | None] = mapped_column(Date, default=None)
    period_end: Mapped[dt.date] = mapped_column(Date, index=True)
    fiscal_year: Mapped[int] = mapped_column(Integer, index=True)
    fiscal_period: Mapped[str] = mapped_column(String(4), index=True)  # FY, Q1..Q4
    accession: Mapped[str | None] = mapped_column(String(32), default=None)
    source: Mapped[str] = mapped_column(String(32), default="sec-edgar-xbrl")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)

    company: Mapped[Company] = relationship(back_populates="facts")


class AnalysisRun(Base):
    """Durable record of one analysis: inputs, outputs, cost, verification."""

    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    question: Mapped[str] = mapped_column(Text)
    tickers: Mapped[str | None] = mapped_column(String(128), default=None)
    answer: Mapped[str | None] = mapped_column(Text, default=None)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    verification: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trace: Mapped[list[Any]] = mapped_column(JSON, default=list)
    model: Mapped[str] = mapped_column(String(64), default="")
    engine: Mapped[str] = mapped_column(String(24), default="anthropic")
    settings_fingerprint: Mapped[str] = mapped_column(String(32), default="")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    output_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    cached_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="ok", index=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    suite: Mapped[str] = mapped_column(String(64))
    engine: Mapped[str] = mapped_column(String(24), default="anthropic")
    model: Mapped[str] = mapped_column(String(64), default="")
    settings_fingerprint: Mapped[str] = mapped_column(String(32), default="")
    n_cases: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    results: Mapped[list[Any]] = mapped_column(JSON, default=list)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
