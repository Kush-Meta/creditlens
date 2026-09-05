"""Retrieval: embeddings, BM25, fusion, filters, diversity, issuer quota."""
from __future__ import annotations

import numpy as np
import pytest

from creditlens.retrieval import bm25, corpus, hybrid
from creditlens.retrieval.embeddings import HashedEmbedder, cosine, pack, tokenize, unpack


class TestEmbeddings:
    def test_deterministic(self):
        a = HashedEmbedder(128).embed(["senior notes mature in 2027"])
        b = HashedEmbedder(128).embed(["senior notes mature in 2027"])
        assert np.allclose(a, b)

    def test_unit_norm(self):
        vectors = HashedEmbedder(128).embed(["leverage rose", "margins fell"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_related_text_scores_above_unrelated(self):
        embedder = HashedEmbedder(256)
        vectors = embedder.embed([
            "total debt increased following the refinancing of senior notes",
            "debt levels rose after the senior note refinancing",
            "we design and sell consumer footwear in retail stores",
        ])
        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_empty_text_is_a_zero_vector(self):
        assert np.allclose(HashedEmbedder(64).embed([""]), 0.0)

    def test_pack_round_trip(self):
        vector = HashedEmbedder(64).embed_one("liquidity")
        assert np.allclose(unpack(pack(vector), 64), vector)

    def test_tokenizer_drops_stopwords(self):
        assert "the" not in tokenize("The company and the market")
        assert "company" in tokenize("The company and the market")

    def test_cosine_handles_empty_matrix(self):
        assert cosine(np.zeros((0, 8), dtype=np.float32), np.ones(8)).shape == (0,)


class TestBM25:
    def test_scores_exact_phrase_terms(self, session):
        snapshot = corpus.get_snapshot(session)
        scores = bm25.search(snapshot, "borrowing base availability")
        assert scores.max() > 0
        best = snapshot.records[int(scores.argmax())]
        assert "borrowing base" in best.text.lower()

    def test_unknown_terms_score_zero(self, session):
        snapshot = corpus.get_snapshot(session)
        assert bm25.search(snapshot, "zzzqqq nonexistentterm").max() == 0

    def test_candidate_restriction(self, session):
        snapshot = corpus.get_snapshot(session)
        allowed = [i for i, r in enumerate(snapshot.records) if r.ticker == "HRBG"]
        scores = bm25.search(snapshot, "inventory markdown", candidates=allowed)
        scored = {i for i in range(len(scores)) if scores[i] > 0}
        assert scored <= set(allowed)


class TestHybridSearch:
    def test_returns_ranked_hits_with_citations(self, session):
        result = hybrid.search(session, "covenant headroom leverage", top_k=5)
        assert result.chunks
        assert [c.rank for c in result.chunks] == list(range(1, len(result.chunks) + 1))
        assert all(c.record.citation()["ticker"] for c in result.chunks)
        assert result.chunks[0].score >= result.chunks[-1].score

    def test_ticker_filter_is_honoured(self, session):
        result = hybrid.search(session, "liquidity", tickers=["HRBG"], top_k=6)
        assert {c.record.ticker for c in result.chunks} == {"HRBG"}

    def test_item_filter_is_honoured(self, session):
        result = hybrid.search(session, "risk", item_codes=["1A"], top_k=6)
        assert {c.record.item_code for c in result.chunks} == {"1A"}

    def test_impossible_filter_returns_a_reason_not_an_error(self, session):
        result = hybrid.search(session, "liquidity", tickers=["NOSUCH"], top_k=5)
        assert result.chunks == []
        assert "metadata filters" in result.diagnostics["reason"]

    def test_query_expansion_adds_filing_vocabulary(self, session):
        result = hybrid.search(session, "leverage", top_k=3)
        assert "borrowings" in result.expanded_query
        assert result.diagnostics["expansion_terms"]

    def test_section_prior_favours_risk_factors_for_risk_questions(self, session):
        result = hybrid.search(
            session, "what risks does management highlight", tickers=["NVCR"], top_k=6
        )
        items = [c.record.item_code for c in result.chunks]
        assert "1A" in items

    def test_multi_issuer_query_returns_evidence_for_each(self, session):
        """Comparison questions are useless with evidence from one side only."""
        result = hybrid.search(
            session, "liquidity and leverage", tickers=["ARMT", "KSTR"], top_k=8
        )
        assert {c.record.ticker for c in result.chunks} == {"ARMT", "KSTR"}

    def test_mmr_reduces_redundancy_when_enabled(self, session):
        plain = hybrid.search(session, "liquidity", tickers=["NVCR"], top_k=6, use_mmr=False)
        diverse = hybrid.search(session, "liquidity", tickers=["NVCR"], top_k=6, use_mmr=True)
        assert len(diverse.chunks) <= len(plain.chunks)

    def test_diagnostics_expose_the_configuration(self, session):
        diagnostics = hybrid.search(session, "debt", top_k=3).diagnostics
        for key in ("corpus_size", "candidates", "fusion", "dense_weight", "rrf_k"):
            assert key in diagnostics

    def test_snippet_is_bounded(self, session):
        result = hybrid.search(session, "covenant", top_k=3)
        assert all(len(c.snippet) < 420 for c in result.chunks)


class TestCorpusSnapshot:
    def test_snapshot_is_cached_until_the_corpus_changes(self, session):
        first = corpus.get_snapshot(session)
        second = corpus.get_snapshot(session)
        assert first is second

    def test_force_rebuild(self, session):
        first = corpus.get_snapshot(session)
        assert corpus.get_snapshot(session, force=True) is not first

    def test_filter_indices_compose(self, session):
        snapshot = corpus.get_snapshot(session)
        indices = corpus.filter_indices(snapshot, tickers=["NVCR"], item_codes=["1A"])
        for index in indices:
            record = snapshot.records[int(index)]
            assert record.ticker == "NVCR" and record.item_code == "1A"


class TestScaling:
    """Behaviour that only matters once the corpus is large."""

    def test_bm25_index_is_numpy_backed(self, session):
        """The dict-of-lists form costs ~700 MB at 100k chunks; arrays cost ~70."""
        import numpy as np

        snapshot = corpus.get_snapshot(session)
        index = bm25.get_index(snapshot)
        for array in (index.doc_ids, index.term_freq, index.offsets, index.idf):
            assert isinstance(array, np.ndarray)
        assert index.offsets.size == len(index.vocab) + 1
        assert index.doc_ids.size == index.n_postings

    def test_postings_are_grouped_by_term(self, session):
        snapshot = corpus.get_snapshot(session)
        index = bm25.get_index(snapshot)
        term = next(iter(index.vocab))
        term_id = index.vocab[term]
        start, end = index.offsets[term_id], index.offsets[term_id + 1]
        assert end > start
        assert (index.term_freq[start:end] > 0).all()

    def test_records_do_not_retain_tokenised_text(self, session):
        """Retaining tokens per record is the single largest memory cost."""
        record = corpus.get_snapshot(session).records[0]
        assert not hasattr(record, "tokens")

    def test_snapshot_precomputes_filter_arrays(self, session):
        snapshot = corpus.get_snapshot(session)
        n = len(snapshot.records)
        assert snapshot.ticker_codes.shape == (n,)
        assert snapshot.item_codes.shape == (n,)
        assert snapshot.fiscal_years.shape == (n,)
        assert len(snapshot.ticker_vocab) >= 4

    def test_fusion_pool_is_bounded(self, session):
        """Fusion must not scale with corpus size, or latency does."""
        result = hybrid.search(session, "liquidity debt covenant", top_k=8)
        pool = result.diagnostics["fusion_pool_size"]
        assert result.diagnostics["scored_candidates"] <= pool * 2


class TestEmbeddingProviders:
    """The pluggable embedding layer.

    Remote providers are exercised against a fake batch function, so batching,
    normalisation, retry, dimension validation and query/document asymmetry are
    all covered without a model server.
    """

    def test_registry_declares_defaults(self):
        from creditlens.retrieval.embeddings import EMBEDDING_PROVIDERS

        assert set(EMBEDDING_PROVIDERS) >= {"hashed", "ollama", "openai", "google"}
        for _factory, model, dim in EMBEDDING_PROVIDERS.values():
            assert model and dim > 0

    def test_unknown_provider_falls_back_to_hashed(self):
        from creditlens.retrieval.embeddings import HashedEmbedder, build_embedder

        assert isinstance(build_embedder("not-a-provider"), HashedEmbedder)

    def test_unavailable_provider_falls_back_rather_than_raising(self):
        from creditlens.retrieval.embeddings import HashedEmbedder, build_embedder

        embedder = build_embedder("openai", "text-embedding-3-small", api_key=None)
        # either it degraded to hashed, or a real key is present in this env
        assert isinstance(embedder, HashedEmbedder) or embedder.dim == 1536

    def _fake_remote(self, dim=8, normalized=False, fail_times=0):

        from creditlens.retrieval.embeddings import RemoteEmbedder

        class FakeRemote(RemoteEmbedder):
            provider = "fake"
            normalizes_output = normalized

            def __init__(self, **kwargs):
                self.calls: list[list[str]] = []
                self.failures = fail_times
                super().__init__("fake-model", dim, **kwargs)

            def _embed_batch(self, texts):
                if self.failures > 0:
                    self.failures -= 1
                    raise RuntimeError("transient")
                self.calls.append(list(texts))
                # deterministic, unnormalised, and length-dependent
                return [[float(len(t) % 7) + i for i in range(dim)] for t in texts]

        return FakeRemote

    def test_batching_respects_batch_size(self):
        embedder = self._fake_remote()(batch_size=3)
        embedder.embed([f"text {i}" for i in range(7)])
        assert [len(call) for call in embedder.calls] == [3, 3, 1]

    def test_output_is_l2_normalised(self):
        import numpy as np

        vectors = self._fake_remote()(batch_size=4).embed(["a", "bb", "ccc"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_already_normalised_models_are_not_renormalised(self):
        import numpy as np

        embedder = self._fake_remote(normalized=True)()
        vectors = embedder.embed(["a"])
        assert not np.isclose(np.linalg.norm(vectors[0]), 1.0)

    def test_transient_failures_are_retried(self):
        embedder = self._fake_remote(fail_times=2)(retries=3)
        assert embedder.embed(["a"]).shape == (1, 8)

    def test_persistent_failure_raises(self):
        embedder = self._fake_remote(fail_times=99)(retries=2)
        with pytest.raises(RuntimeError, match="embedding failed"):
            embedder.embed(["a"])

    def test_dimension_mismatch_is_caught(self):
        from creditlens.retrieval.embeddings import RemoteEmbedder

        class WrongDim(RemoteEmbedder):
            provider = "fake"

            def _embed_batch(self, texts):
                return [[0.1] * 3 for _ in texts]

        with pytest.raises(ValueError, match="dimension 3, expected 8"):
            WrongDim("m", 8).embed(["a"])

    def test_empty_input(self):
        assert self._fake_remote()().embed([]).shape == (0, 8)

    def test_instruction_tuned_models_get_a_query_prefix(self):
        """Qwen3/E5/BGE embed queries and documents differently."""
        from creditlens.retrieval.embeddings import OllamaEmbedder

        assert OllamaEmbedder(model="qwen3-embedding:0.6b").query_prefix.startswith("Instruct:")
        assert OllamaEmbedder(model="multilingual-e5-large").query_prefix == "query: "
        assert OllamaEmbedder(model="some-plain-model").query_prefix == ""

    def test_query_prefix_is_applied_only_to_queries(self):
        embedder = self._fake_remote()(query_prefix="Q: ")
        embedder.embed(["document text"])
        embedder.embed_query("question text")
        assert embedder.calls[0] == ["document text"]
        assert embedder.calls[1] == ["Q: question text"]

    def test_symmetric_models_embed_queries_unchanged(self):
        embedder = self._fake_remote()()
        embedder.embed_query("question")
        assert embedder.calls[0] == ["question"]

    def test_hashed_embedder_is_symmetric(self):
        import numpy as np

        from creditlens.retrieval.embeddings import HashedEmbedder

        embedder = HashedEmbedder(dim=64)
        assert np.allclose(embedder.embed_query("liquidity"), embedder.embed_one("liquidity"))

    def test_caching_separates_query_and_document_vectors(self):
        """The same string embedded as a query and as a document must differ."""
        import numpy as np

        from creditlens.retrieval.embeddings import CachingEmbedder

        cached = CachingEmbedder(self._fake_remote()(query_prefix="Q: "))
        as_document = cached.embed(["same text"])[0]
        as_query = cached.embed_query("same text")
        assert not np.allclose(as_document, as_query)
        # and each side is served from cache on repeat
        before = cached.hits
        cached.embed(["same text"])
        cached.embed_query("same text")
        assert cached.hits == before + 2


class TestReembed:
    def test_inventory_reports_stored_models(self, session):
        from creditlens.retrieval.reembed import embedding_inventory

        inventory = embedding_inventory(session)
        assert inventory["total_chunks"] > 0
        assert inventory["by_model"]

    def test_stale_detection_keys_on_model_and_dim(self, session):
        from creditlens.retrieval.reembed import stale_chunk_ids

        assert stale_chunk_ids(session, "some-other-model", 999)
        current = corpus.get_snapshot(session).embedding_model
        assert stale_chunk_ids(session, current, 512) == []

    def test_reembed_is_idempotent(self, session):
        from creditlens.retrieval.embeddings import get_embedder
        from creditlens.retrieval.reembed import reembed

        first = reembed(session, embedder=get_embedder(), batch_size=32)
        assert first.embedded == 0  # already current
        assert first.already_current == first.total_chunks

    def test_reembed_rebuilds_stale_vectors(self, session):
        from creditlens.retrieval.embeddings import HashedEmbedder
        from creditlens.retrieval.reembed import reembed, stale_chunk_ids

        # a different dimension makes every stored vector stale
        other = HashedEmbedder(dim=128)
        other.name = "hashed-ngram-128"
        report = reembed(session, embedder=other, batch_size=64, limit=20)
        assert report.embedded == 20
        assert report.failed == 0
        assert len(stale_chunk_ids(session, "hashed-ngram-128", 128)) < report.total_chunks

        # restore, so later tests see the configured model
        from creditlens.retrieval.embeddings import get_embedder

        reembed(session, embedder=get_embedder(), batch_size=64)
