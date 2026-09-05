"""Okapi BM25 over the corpus snapshot, backed by numpy.

Lexical matching is not optional in this domain: analysts search for exact
strings - "senior notes", "revolving credit facility", a covenant name. A
pure-embedding retriever misses those reliably, which is why the hybrid layer
fuses BM25 with dense scores instead of choosing one.

**Why the index is flat numpy arrays rather than `dict[str, list[tuple]]`.**
The dict-of-lists form is the obvious implementation and it does not survive
scale: at ~100k chunks it costs ~718 MB for the postings alone, plus ~844 MB
if tokenised documents are retained on the records. Storing postings as three
parallel arrays in term order - doc ids, term frequencies, and per-term offsets
into both - holds the same index in ~120 MB and makes scoring a vectorised
slice per query term rather than a Python loop over postings.

The build is single-pass into typed `array.array` buffers, then one argsort
into term order, so peak build memory stays near the size of the final index
instead of several times it.
"""
from __future__ import annotations

import array
from dataclasses import dataclass

import numpy as np

from creditlens.observability import get_logger
from creditlens.retrieval.corpus import Snapshot
from creditlens.retrieval.embeddings import tokenize

log = get_logger(__name__)

K1 = 1.5
B = 0.75


@dataclass
class BM25Index:
    """Postings in term order: term `t` owns rows `offsets[t]:offsets[t+1]`."""

    version: tuple[int, int]
    n_docs: int
    doc_len: np.ndarray      # float32[n_docs]
    avg_len: float
    vocab: dict[str, int]    # term -> term id
    doc_ids: np.ndarray      # int32[n_postings], grouped by term
    term_freq: np.ndarray    # float32[n_postings]
    offsets: np.ndarray      # int64[n_terms + 1]
    idf: np.ndarray          # float32[n_terms]

    @property
    def n_postings(self) -> int:
        return int(self.doc_ids.size)

    def nbytes(self) -> int:
        return int(
            self.doc_len.nbytes + self.doc_ids.nbytes + self.term_freq.nbytes
            + self.offsets.nbytes + self.idf.nbytes
        )


_INDEX: BM25Index | None = None


def build(snapshot: Snapshot) -> BM25Index:
    n_docs = len(snapshot.records)
    doc_len = np.zeros(n_docs, dtype=np.float32)

    vocab: dict[str, int] = {}
    term_ids = array.array("i")
    doc_ids = array.array("i")
    term_freq = array.array("f")

    for doc, record in enumerate(snapshot.records):
        # Tokenised text is used here and discarded; keeping it on the record
        # is what makes a large corpus unaffordable.
        tokens = tokenize(record.text)
        doc_len[doc] = len(tokens)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for term, freq in counts.items():
            term_id = vocab.get(term)
            if term_id is None:
                term_id = len(vocab)
                vocab[term] = term_id
            term_ids.append(term_id)
            doc_ids.append(doc)
            term_freq.append(float(freq))

    term_id_array = np.frombuffer(term_ids, dtype=np.int32)
    doc_id_array = np.frombuffer(doc_ids, dtype=np.int32)
    freq_array = np.frombuffer(term_freq, dtype=np.float32)

    order = np.argsort(term_id_array, kind="stable")
    sorted_terms = term_id_array[order]
    sorted_docs = np.ascontiguousarray(doc_id_array[order])
    sorted_freqs = np.ascontiguousarray(freq_array[order])

    n_terms = len(vocab)
    counts_per_term = np.bincount(sorted_terms, minlength=n_terms)
    offsets = np.zeros(n_terms + 1, dtype=np.int64)
    np.cumsum(counts_per_term, out=offsets[1:])

    # Okapi IDF, computed once per term from its document frequency.
    document_frequency = counts_per_term.astype(np.float64)
    idf = np.log(
        1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)

    index = BM25Index(
        version=snapshot.version,
        n_docs=n_docs,
        doc_len=doc_len,
        avg_len=float(doc_len.mean()) if n_docs else 1.0,
        vocab=vocab,
        doc_ids=sorted_docs,
        term_freq=sorted_freqs,
        offsets=offsets,
        idf=idf,
    )
    log.info("bm25 index built", extra={
        "docs": n_docs, "terms": n_terms,
        "postings": index.n_postings, "index_mb": round(index.nbytes() / 1e6, 1),
    })
    return index


def get_index(snapshot: Snapshot) -> BM25Index:
    global _INDEX
    if _INDEX is None or _INDEX.version != snapshot.version:
        _INDEX = build(snapshot)
    return _INDEX


def invalidate() -> None:
    global _INDEX
    _INDEX = None


def search(
    snapshot: Snapshot,
    query: str,
    *,
    candidates: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """BM25 score for every document, zero outside the candidate set."""
    index = get_index(snapshot)
    scores = np.zeros(index.n_docs, dtype=np.float32)
    if index.n_docs == 0:
        return scores

    terms = tokenize(query)
    if not terms:
        return scores

    # Length normalisation is document-wide and identical for every term.
    norm = (1 - B) + B * (index.doc_len / (index.avg_len or 1.0))

    for term in terms:
        term_id = index.vocab.get(term)
        if term_id is None:
            continue
        start, end = index.offsets[term_id], index.offsets[term_id + 1]
        if start == end:
            continue
        docs = index.doc_ids[start:end]
        freqs = index.term_freq[start:end]
        contribution = index.idf[term_id] * (freqs * (K1 + 1)) / (
            freqs + K1 * norm[docs]
        )
        # A term may appear once per document in the postings, so scatter-add
        # is not required for correctness - but duplicate query terms make it
        # the safe accumulation.
        np.add.at(scores, docs, contribution)

    if candidates is not None:
        mask = np.zeros(index.n_docs, dtype=bool)
        mask[np.asarray(candidates, dtype=np.int64)] = True
        scores[~mask] = 0.0
    return scores
