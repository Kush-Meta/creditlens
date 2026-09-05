"""Rebuild stored vectors under a different embedding model.

Changing embedding model changes the vector space, and two spaces mixed in one
index degrade retrieval in a way no metric attributes to the cause. So the
migration is explicit, resumable, and reports what it did:

* chunks already embedded with the target model are skipped, so an interrupted
  run continues rather than starting over;
* work is committed in batches, so an interruption loses at most one batch;
* the old vector is replaced only after the new one is computed, so a failure
  never leaves a chunk with no vector at all.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from creditlens.db.models import Chunk, Embedding
from creditlens.observability import get_logger
from creditlens.retrieval import bm25 as bm25_mod
from creditlens.retrieval import corpus as corpus_mod
from creditlens.retrieval.embeddings import Embedder, get_embedder, pack

log = get_logger(__name__)


@dataclass
class ReembedReport:
    model: str
    dim: int
    total_chunks: int = 0
    already_current: int = 0
    embedded: int = 0
    failed: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.embedded / self.duration_s if self.duration_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dim": self.dim,
            "total_chunks": self.total_chunks,
            "already_current": self.already_current,
            "embedded": self.embedded,
            "failed": self.failed,
            "duration_s": round(self.duration_s, 1),
            "chunks_per_second": round(self.rate, 1),
            "errors": self.errors[:10],
        }


def stale_chunk_ids(session: Session, model: str, dim: int) -> list[int]:
    """Chunks whose stored vector was not produced by the target model."""
    rows = session.execute(
        select(Chunk.id, Embedding.model, Embedding.dim)
        .outerjoin(Embedding, Embedding.chunk_id == Chunk.id)
        .order_by(Chunk.id)
    ).all()
    return [
        chunk_id for chunk_id, stored_model, stored_dim in rows
        if stored_model != model or stored_dim != dim
    ]


def reembed(
    session: Session,
    *,
    embedder: Embedder | None = None,
    batch_size: int = 64,
    limit: int | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> ReembedReport:
    embedder = embedder or get_embedder()
    inner = getattr(embedder, "inner", embedder)
    model_name = getattr(inner, "name", embedder.name)

    total = session.scalar(select(func.count(Chunk.id))) or 0
    stale = stale_chunk_ids(session, model_name, embedder.dim)
    if limit:
        stale = stale[:limit]

    report = ReembedReport(
        model=model_name, dim=embedder.dim,
        total_chunks=total, already_current=total - len(stale),
    )
    if not stale:
        log.info("vectors already current", extra={"model": model_name, "chunks": total})
        return report

    log.info("re-embedding", extra={
        "model": model_name, "dim": embedder.dim,
        "stale": len(stale), "total": total, "batch_size": batch_size,
    })
    started = time.perf_counter()

    for offset in range(0, len(stale), batch_size):
        window = stale[offset:offset + batch_size]
        chunks = list(session.scalars(select(Chunk).where(Chunk.id.in_(window))))
        if not chunks:
            continue
        try:
            vectors = embedder.embed([chunk.text for chunk in chunks])
        except Exception as exc:
            report.failed += len(chunks)
            report.errors.append(f"chunks {window[0]}-{window[-1]}: {exc}")
            log.warning("batch failed; continuing", extra={"error": str(exc)[:200]})
            continue

        existing = {
            row.chunk_id: row for row in session.scalars(
                select(Embedding).where(Embedding.chunk_id.in_([c.id for c in chunks]))
            )
        }
        for index, chunk in enumerate(chunks):
            blob = pack(vectors[index])
            row = existing.get(chunk.id)
            if row is None:
                session.add(Embedding(
                    chunk_id=chunk.id, model=model_name, dim=embedder.dim, vector=blob
                ))
            else:
                row.model, row.dim, row.vector = model_name, embedder.dim, blob
        # Commit per batch: an interruption costs one batch, not the whole run.
        session.commit()
        report.embedded += len(chunks)
        if progress:
            elapsed = time.perf_counter() - started
            progress(report.embedded, len(stale), report.embedded / elapsed if elapsed else 0.0)

    report.duration_s = time.perf_counter() - started
    corpus_mod.invalidate()
    bm25_mod.invalidate()
    log.info("re-embed complete", extra=report.to_dict())
    return report


def embedding_inventory(session: Session) -> dict[str, Any]:
    """What models the stored vectors came from - surfaced by /health."""
    rows = session.execute(
        select(Embedding.model, Embedding.dim, func.count(Embedding.chunk_id))
        .group_by(Embedding.model, Embedding.dim)
    ).all()
    total_chunks = session.scalar(select(func.count(Chunk.id))) or 0
    embedded = sum(count for _, _, count in rows)
    return {
        "total_chunks": total_chunks,
        "embedded": embedded,
        "unembedded": total_chunks - embedded,
        "by_model": [
            {"model": model, "dim": dim, "chunks": count} for model, dim, count in rows
        ],
    }
