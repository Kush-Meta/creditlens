"""Hybrid retrieval: BM25 + dense, fused with RRF, diversified with MMR.

Pipeline
--------
1. **Metadata pre-filter** - ticker/form/year/item. Most analyst questions are
   scoped ("the latest 10-K", "Ford"), and filtering before scoring is both
   faster and far more precise than hoping the ranker infers scope.
2. **Query expansion** - credit vocabulary rarely matches filing vocabulary
   ("leverage" vs "borrowings", "liquidity" vs "revolving credit facility").
   A small curated map closes that gap for the lexical leg.
3. **Two rankers, then Reciprocal Rank Fusion.** RRF fuses by rank, not score,
   so BM25's unbounded scores and cosine's [-1,1] never need calibrating -
   the classic failure mode of naive score-weighted hybrid search.
4. **Section priors** - a question about risks should prefer Item 1A over the
   boilerplate in Item 5. Applied as a small multiplicative prior, never large
   enough to override a strong direct match.
5. **MMR** - filings repeat themselves across quarters; without diversity the
   top-k is often five near-identical paragraphs from the same section.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from creditlens.config import get_settings
from creditlens.observability import METRICS, get_logger
from creditlens.retrieval import bm25, corpus
from creditlens.retrieval.corpus import ChunkRecord, Snapshot
from creditlens.retrieval.embeddings import get_embedder

log = get_logger(__name__)

#: credit-analyst term -> filing vocabulary. Expansion is additive and only
#: affects the lexical leg, so a bad expansion cannot hijack the dense ranking.
EXPANSIONS: dict[str, tuple[str, ...]] = {
    "leverage": ("debt", "borrowings", "indebtedness", "notes payable"),
    "liquidity": ("cash", "credit facility", "revolving", "commercial paper", "liquidity"),
    "covenant": ("covenants", "indenture", "financial covenants", "default"),
    "risk": ("risk factors", "adversely", "uncertainties", "could harm"),
    "risks": ("risk factors", "adversely", "uncertainties"),
    "margin": ("gross margin", "operating margin", "cost of revenue", "profitability"),
    "margins": ("gross margin", "operating margin", "cost of revenue"),
    "credit quality": ("credit ratings", "debt", "leverage", "liquidity"),
    "creditworthiness": ("credit ratings", "debt", "liquidity", "cash flow"),
    "maturity": ("maturities", "due", "repayment", "refinance"),
    "refinancing": ("refinance", "maturities", "senior notes", "repay"),
    "guidance": ("outlook", "expect", "anticipate"),
    "downturn": ("recession", "macroeconomic", "demand weakness"),
    "rating": ("credit ratings", "downgrade", "investment grade"),
    "buyback": ("repurchase", "share repurchase", "treasury stock"),
    "dividend": ("dividends", "distributions", "capital return"),
    "headcount": ("employees", "restructuring", "workforce"),
    "supply chain": ("suppliers", "component", "shortages", "logistics"),
}

#: keyword -> preferred 10-K/10-Q item, applied as a prior on the fused score
SECTION_PRIORS: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\brisk|threat|headwind|adverse|litigation|regulat", re.I), "1A", 1.25),
    (re.compile(r"\bmargin|revenue|growth|driver|caused|why|performance|expense", re.I), "7", 1.20),
    (re.compile(r"\bliquidity|cash|facility|maturit|refinanc|covenant|debt", re.I), "7", 1.15),
    (re.compile(r"\bmarket risk|interest rate|foreign currency|hedg", re.I), "7A", 1.25),
    (re.compile(r"\bbusiness|segment|product|competitor|strategy", re.I), "1", 1.12),
    (re.compile(r"\blegal|lawsuit|proceeding", re.I), "3", 1.20),
)


@dataclass
class RetrievedChunk:
    record: ChunkRecord
    rank: int
    score: float
    dense_score: float
    lexical_score: float
    dense_rank: int | None
    lexical_rank: int | None
    section_prior: float
    snippet: str
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "rank": self.rank,
            "chunk_id": self.record.chunk_id,
            "score": round(self.score, 5),
            "scores": {
                "dense": round(self.dense_score, 5),
                "lexical": round(self.lexical_score, 5),
                "dense_rank": self.dense_rank,
                "lexical_rank": self.lexical_rank,
                "section_prior": round(self.section_prior, 3),
            },
            "matched_terms": self.matched_terms[:8],
            "citation": self.record.citation(),
            "snippet": self.snippet,
        }
        if include_text:
            payload["text"] = self.record.text
        return payload


@dataclass
class RetrievalResult:
    query: str
    expanded_query: str
    chunks: list[RetrievedChunk]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        return {
            "query": self.query,
            "expanded_query": self.expanded_query,
            "hits": [c.to_dict(include_text) for c in self.chunks],
            "diagnostics": self.diagnostics,
        }


def expand_query(query: str) -> tuple[str, list[str]]:
    lower = query.lower()
    added: list[str] = []
    for term, expansions in EXPANSIONS.items():
        if term in lower:
            for expansion in expansions:
                if expansion not in lower:
                    added.append(expansion)
    return (query + " " + " ".join(added)).strip(), added


def section_prior(query: str, record: ChunkRecord) -> float:
    prior = 1.0
    item = (record.item_code or "").upper()
    for pattern, wanted_item, boost in SECTION_PRIORS:
        if pattern.search(query) and item == wanted_item:
            prior = max(prior, boost)
    return prior


def _leg_ranks(
    scores: np.ndarray, candidates: np.ndarray, pool_size: int
) -> dict[int, int]:
    """1-based ranks for one ranker, restricted to its top `pool_size` hits.

    Ranking every candidate was affordable at a few thousand chunks and is not
    at eighty thousand: building and sorting a Python list of that many tuples
    dominated query latency. Reciprocal rank fusion only ever gives meaningful
    weight to the head of each list - an item ranked 10,000th contributes
    1/(60+10000) - so partitioning to the head first is faster and changes
    results only below the noise floor.
    """
    subset = scores[candidates]
    positive = np.nonzero(subset > 0)[0]
    if positive.size == 0:
        return {}
    if positive.size > pool_size:
        head = np.argpartition(-subset[positive], pool_size - 1)[:pool_size]
        positive = positive[head]
    order = positive[np.argsort(-subset[positive], kind="stable")]
    return {int(candidates[position]): rank for rank, position in enumerate(order, start=1)}


def _mmr(
    order: list[int],
    vectors: np.ndarray,
    relevance: dict[int, float],
    top_k: int,
    lam: float,
) -> list[int]:
    selected: list[int] = []
    pool = list(order)
    while pool and len(selected) < top_k:
        best_idx, best_val = None, -1e9
        for candidate in pool[: max(top_k * 6, 30)]:
            rel = relevance.get(candidate, 0.0)
            if selected:
                sims = vectors[selected] @ vectors[candidate]
                redundancy = float(np.max(sims))
            else:
                redundancy = 0.0
            value = lam * rel - (1 - lam) * redundancy
            if value > best_val:
                best_val, best_idx = value, candidate
        if best_idx is None:
            break
        selected.append(best_idx)
        pool.remove(best_idx)
    return selected


def _apply_issuer_quota(
    selected: list[int],
    ordered: list[int],
    snapshot: Snapshot,
    tickers: Sequence[str],
    top_k: int,
    floor: int,
) -> list[int]:
    """Guarantee every scoped issuer a minimum share of the result slots.

    A comparison question ("compare Oracle and Microsoft's liquidity") is
    useless if all eight passages come from one issuer, which is exactly what
    a global ranking produces when one company's filings phrase the topic more
    strongly. Measured on the labelled comparison cases, this is the
    difference between an answer with evidence for both sides and one with
    evidence for neither.
    """
    wanted = list(dict.fromkeys(t.upper() for t in tickers))
    quota = max(1, min(floor, top_k // len(wanted)))
    result = list(selected)

    for ticker in wanted:
        present = [i for i in result if snapshot.records[i].ticker == ticker]
        if len(present) >= quota:
            continue
        candidates = [
            i for i in ordered
            if snapshot.records[i].ticker == ticker and i not in result
        ]
        needed = quota - len(present)
        for candidate in candidates[:needed]:
            # displace the lowest-ranked passage from the most over-represented
            # issuer, never one that is already at its floor
            counts: dict[str, int] = {}
            for i in result:
                counts[snapshot.records[i].ticker] = counts.get(
                    snapshot.records[i].ticker, 0
                ) + 1
            donor = max(counts, key=lambda t: counts[t])
            if counts[donor] <= quota:
                break
            for i in reversed(result):
                if snapshot.records[i].ticker == donor:
                    result.remove(i)
                    break
            result.append(candidate)

    return sorted(result, key=lambda i: ordered.index(i) if i in ordered else 10**6)


def _snippet(text: str, terms: Sequence[str], width: int = 320) -> str:
    lower = text.lower()
    position = -1
    for term in terms:
        position = lower.find(term.lower())
        if position >= 0:
            break
    if position < 0:
        return text[:width].strip() + ("..." if len(text) > width else "")
    start = max(0, position - width // 3)
    end = min(len(text), start + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def search(
    session: Session,
    query: str,
    *,
    tickers: Sequence[str] | None = None,
    form_types: Sequence[str] | None = None,
    item_codes: Sequence[str] | None = None,
    fiscal_years: Sequence[int] | None = None,
    top_k: int | None = None,
    use_mmr: bool | None = None,
) -> RetrievalResult:
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    use_mmr = settings.retrieval_use_mmr if use_mmr is None else use_mmr
    snapshot: Snapshot = corpus.get_snapshot(session)

    if len(snapshot) == 0:
        return RetrievalResult(query, query, [], {"corpus_size": 0, "reason": "empty corpus"})

    candidates = corpus.filter_indices(
        snapshot,
        tickers=tickers, form_types=form_types,
        item_codes=item_codes, fiscal_years=fiscal_years,
    )
    if candidates.size == 0:
        return RetrievalResult(query, query, [], {
            "corpus_size": len(snapshot),
            "candidates": 0,
            "reason": "no documents matched the metadata filters",
            "filters": {
                "tickers": list(tickers or []), "form_types": list(form_types or []),
                "item_codes": list(item_codes or []), "fiscal_years": list(fiscal_years or []),
            },
        })

    expanded, added_terms = expand_query(query)

    embedder = get_embedder()
    # embed_query, not embed_one: instruction-tuned models embed the query side
    # differently from the document side, and using the document path here costs
    # measurable recall
    query_vector = embedder.embed_query(query)
    dense_scores = np.full(len(snapshot), -1.0, dtype=np.float32)
    dense_scores[candidates] = snapshot.vectors[candidates] @ query_vector

    lexical_scores = bm25.search(snapshot, expanded, candidates=candidates)

    pool_size = max(settings.retrieval_candidate_k, top_k * 8, 100)
    dense_ranks = _leg_ranks(dense_scores, candidates, pool_size)
    lexical_ranks = _leg_ranks(lexical_scores, candidates, pool_size)

    rrf_k = settings.rrf_k
    weight_dense = settings.dense_weight
    fused: dict[int, float] = {}
    # Fusion, section priors and everything downstream run over the union of
    # the two heads - a few hundred items - not the whole candidate set.
    for idx in set(dense_ranks) | set(lexical_ranks):
        score = 0.0
        if idx in dense_ranks:
            score += weight_dense / (rrf_k + dense_ranks[idx])
        if idx in lexical_ranks:
            score += (1 - weight_dense) / (rrf_k + lexical_ranks[idx])
        if score > 0:
            score *= section_prior(query, snapshot.records[idx])
            fused[idx] = score

    if not fused:
        return RetrievalResult(query, expanded, [], {
            "corpus_size": len(snapshot),
            "candidates": int(candidates.size),
            "reason": "no lexical or dense signal above zero",
        })

    ordered = sorted(fused, key=lambda i: -fused[i])[: settings.retrieval_candidate_k]
    if use_mmr and len(ordered) > top_k:
        selected = _mmr(ordered, snapshot.vectors, fused, top_k, settings.mmr_lambda)
    else:
        selected = ordered[:top_k]
    if tickers and len({t.upper() for t in tickers}) > 1:
        selected = _apply_issuer_quota(
            selected, ordered, snapshot, tickers, top_k, settings.per_issuer_floor
        )

    query_terms = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", query)
    hits: list[RetrievedChunk] = []
    for rank, idx in enumerate(selected, start=1):
        record = snapshot.records[idx]
        matched = [t for t in query_terms if t.lower() in record.text.lower()]
        hits.append(RetrievedChunk(
            record=record,
            rank=rank,
            score=float(fused[idx]),
            dense_score=float(dense_scores[idx]),
            lexical_score=float(lexical_scores[idx]),
            dense_rank=dense_ranks.get(idx),
            lexical_rank=lexical_ranks.get(idx),
            section_prior=section_prior(query, record),
            snippet=_snippet(record.text, matched or query_terms),
            matched_terms=matched,
        ))

    METRICS.inc("creditlens_retrievals_total")
    METRICS.observe("creditlens_retrieval_hits", len(hits))

    return RetrievalResult(query, expanded, hits, {
        "corpus_size": len(snapshot),
        "candidates": int(candidates.size),
        "expansion_terms": added_terms,
        "embedding_model": snapshot.embedding_model,
        "fusion": "reciprocal rank fusion",
        "rrf_k": rrf_k,
        "dense_weight": weight_dense,
        "mmr": bool(use_mmr and len(ordered) > top_k),
        "mmr_lambda": settings.mmr_lambda,
        "scored_candidates": len(fused),
        "fusion_pool_size": pool_size,
    })
