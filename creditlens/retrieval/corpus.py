"""In-memory corpus snapshot shared by the lexical and dense retrievers.

Loading chunk rows once and holding them as parallel arrays (rather than
re-querying per search) is what keeps end-to-end retrieval in the low
milliseconds at this corpus size. The snapshot is invalidated by a cheap
(count, max_id) probe, so ingestion is picked up without a restart.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from creditlens.db.models import Chunk, Company, Embedding, Filing
from creditlens.observability import get_logger
from creditlens.retrieval.embeddings import get_embedder, unpack

log = get_logger(__name__)

#: Above this many missing vectors, refuse to embed inline and require a
#: migration instead. Sized so a handful of newly ingested chunks still work.
INLINE_EMBED_LIMIT = 500


@dataclass
class ChunkRecord:
    chunk_id: int
    company_id: int
    ticker: str
    company_name: str
    filing_id: int
    accession: str
    form_type: str
    fiscal_year: int | None
    fiscal_period: str | None
    filing_date: str
    url: str | None
    item_code: str | None
    section_title: str | None
    heading_path: str | None
    ordinal: int
    text: str
    token_count: int
    is_synthetic: bool

    def citation(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "ticker": self.ticker,
            "company": self.company_name,
            "form_type": self.form_type,
            "period": f"{self.fiscal_period or ''}{self.fiscal_year or ''}".strip(),
            "filing_date": self.filing_date,
            "accession": self.accession,
            "item": self.item_code,
            "section": self.section_title,
            "url": self.url,
            "is_synthetic": self.is_synthetic,
        }


@dataclass
class Snapshot:
    version: tuple[int, int]
    records: list[ChunkRecord]
    vectors: np.ndarray
    embedding_model: str
    by_id: dict[int, int] = field(default_factory=dict)
    #: metadata as parallel numpy arrays. Filtering 87k records through a
    #: Python generator costs milliseconds per query and grows linearly with
    #: the corpus; integer-coded arrays make every filter a vectorised compare.
    ticker_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    item_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    form_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    fiscal_years: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    ticker_vocab: dict[str, int] = field(default_factory=dict)
    item_vocab: dict[str, int] = field(default_factory=dict)
    form_vocab: dict[str, int] = field(default_factory=dict)
    #: chunks whose stored vector does not match the configured model
    stale_vectors: int = 0

    def __len__(self) -> int:
        return len(self.records)

    def codes_for(self, vocab: dict[str, int], values: Sequence[str]) -> np.ndarray:
        wanted = [vocab[v.upper()] for v in values if v.upper() in vocab]
        return np.asarray(wanted, dtype=np.int32)


_lock = threading.Lock()
_snapshot: Snapshot | None = None


def _version(session: Session) -> tuple[int, int]:
    row = session.execute(
        select(func.count(Chunk.id), func.coalesce(func.max(Chunk.id), 0))
    ).one()
    return int(row[0]), int(row[1])


def get_snapshot(session: Session, *, force: bool = False) -> Snapshot:
    global _snapshot
    version = _version(session)
    with _lock:
        if _snapshot is not None and _snapshot.version == version and not force:
            return _snapshot
        snapshot = _build(session, version)
        _snapshot = snapshot
        return snapshot


def invalidate() -> None:
    global _snapshot
    with _lock:
        _snapshot = None


def _build(session: Session, version: tuple[int, int]) -> Snapshot:
    embedder = get_embedder()
    rows = session.execute(
        select(Chunk, Filing, Company, Embedding)
        .join(Filing, Chunk.filing_id == Filing.id)
        .join(Company, Chunk.company_id == Company.id)
        .outerjoin(Embedding, Embedding.chunk_id == Chunk.id)
        .order_by(Chunk.id)
    ).all()

    records: list[ChunkRecord] = []
    vectors = np.zeros((len(rows), embedder.dim), dtype=np.float32)
    missing_vectors: list[int] = []

    for i, (chunk, filing, company, embedding) in enumerate(rows):
        records.append(ChunkRecord(
            chunk_id=chunk.id,
            company_id=company.id,
            ticker=company.ticker,
            company_name=company.name,
            filing_id=filing.id,
            accession=filing.accession,
            form_type=filing.form_type,
            fiscal_year=filing.fiscal_year,
            fiscal_period=filing.fiscal_period,
            filing_date=filing.filing_date.isoformat(),
            url=filing.url,
            item_code=chunk.item_code,
            section_title=chunk.section_title,
            heading_path=chunk.heading_path,
            ordinal=chunk.ordinal,
            text=chunk.text,
            token_count=chunk.token_count,
            is_synthetic=filing.is_synthetic or company.is_synthetic,
        ))
        if embedding is not None and embedding.dim == embedder.dim:
            vectors[i] = unpack(embedding.vector, embedding.dim)
        else:
            missing_vectors.append(i)

    if missing_vectors:
        # Embedding on demand is fine for a handful of stragglers and ruinous at
        # corpus scale: with a real model, re-embedding 76k chunks inline would
        # block a request for over an hour. Above the threshold we leave the
        # vectors zeroed - lexical retrieval still works - and say loudly that a
        # migration is required, rather than silently hanging or silently
        # serving a half-populated index.
        if len(missing_vectors) <= INLINE_EMBED_LIMIT:
            fresh = embedder.embed([records[i].text for i in missing_vectors])
            for slot, i in enumerate(missing_vectors):
                vectors[i] = fresh[slot]
            log.info("embedded stragglers on the fly",
                     extra={"count": len(missing_vectors)})
        else:
            log.warning(
                "vectors missing for the configured embedding model; dense "
                "retrieval is degraded until `creditlens reembed` is run",
                extra={
                    "missing": len(missing_vectors), "total": len(records),
                    "model": embedder.name, "dim": embedder.dim,
                },
            )

    ticker_vocab: dict[str, int] = {}
    item_vocab: dict[str, int] = {}
    form_vocab: dict[str, int] = {}

    def code(vocab: dict[str, int], value: str) -> int:
        key = (value or "").upper()
        if key not in vocab:
            vocab[key] = len(vocab)
        return vocab[key]

    snapshot = Snapshot(
        stale_vectors=len(missing_vectors),
        version=version,
        records=records,
        vectors=vectors,
        embedding_model=embedder.name,
        by_id={r.chunk_id: i for i, r in enumerate(records)},
        ticker_codes=np.fromiter(
            (code(ticker_vocab, r.ticker) for r in records), np.int32, len(records)),
        item_codes=np.fromiter(
            (code(item_vocab, r.item_code or "") for r in records), np.int32, len(records)),
        form_codes=np.fromiter(
            (code(form_vocab, r.form_type) for r in records), np.int32, len(records)),
        fiscal_years=np.fromiter(
            (r.fiscal_year or -1 for r in records), np.int32, len(records)),
        ticker_vocab=ticker_vocab,
        item_vocab=item_vocab,
        form_vocab=form_vocab,
    )
    log.info(
        "corpus snapshot built",
        extra={"chunks": len(records), "version": list(version), "model": embedder.name},
    )
    return snapshot


def filter_indices(
    snapshot: Snapshot,
    *,
    tickers: Sequence[str] | None = None,
    form_types: Sequence[str] | None = None,
    item_codes: Sequence[str] | None = None,
    fiscal_years: Sequence[int] | None = None,
    accessions: Sequence[str] | None = None,
) -> np.ndarray:
    """Metadata pre-filter. Returns positional indices into the snapshot."""
    n = len(snapshot.records)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    mask = np.ones(n, dtype=bool)

    if tickers:
        mask &= np.isin(snapshot.ticker_codes,
                        snapshot.codes_for(snapshot.ticker_vocab, tickers))
    if form_types:
        mask &= np.isin(snapshot.form_codes,
                        snapshot.codes_for(snapshot.form_vocab, form_types))
    if item_codes:
        mask &= np.isin(snapshot.item_codes,
                        snapshot.codes_for(snapshot.item_vocab, item_codes))
    if fiscal_years:
        mask &= np.isin(snapshot.fiscal_years,
                        np.asarray([int(y) for y in fiscal_years], dtype=np.int32))
    if accessions:
        # rare, and not worth a code array of its own
        wanted = set(accessions)
        mask &= np.fromiter(
            (r.accession in wanted for r in snapshot.records), bool, n)
    return np.nonzero(mask)[0]
