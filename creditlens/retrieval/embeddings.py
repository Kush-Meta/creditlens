"""Embedding backends behind one interface.

Three kinds of backend, all satisfying `Embedder`:

* **`hashed`** - a deterministic hashed-ngram model with no dependencies. It is
  reproducible, offline, needs no key, and makes the *architecture* the thing
  under test in CI. It is also, measurably, a lexical model in disguise:
  paraphrase queries collapse to ~0.1 nDCG. It is the fallback, not the goal.
* **`ollama`** - a real embedding model served locally. No key, no per-token
  cost, and no data leaving the machine, which matters for a tool that reads
  financial filings.
* **`openai` / `google`** - hosted embedding models, for when quality matters
  more than locality.

Swapping backends changes the vector space, so stored vectors from a different
model are not comparable. Every row records the model that produced it, the
settings fingerprint includes it, and `creditlens reembed` is the migration
path. Silently mixing two embedding spaces would degrade retrieval in a way
that no metric would obviously attribute to the cause.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from creditlens.observability import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-\.]*")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are",
    "was", "were", "be", "been", "as", "at", "by", "that", "this", "it", "with",
    "from", "we", "our", "its", "their", "which", "such", "may", "will", "has",
    "have", "had", "not", "no", "but", "if", "than", "these", "those", "also",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _bigrams(tokens: Sequence[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized row vectors."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query.

        Separate from `embed` because instruction-tuned embedding models
        (Qwen3-Embedding, E5, BGE and friends) are trained asymmetrically: the
        document side is embedded raw, the query side with a task instruction.
        Embedding both sides identically silently costs recall - measurably so
        here - and the symmetric default keeps that detail invisible for models
        that do not need it.
        """
        return self.embed_one(text)


class HashedEmbedder(Embedder):
    """Signed feature hashing over unigrams and bigrams."""

    def __init__(self, dim: int = 512, seed: int = 17):
        self.dim = dim
        self.seed = seed
        self.name = f"hashed-ngram-{dim}"
        self._cache: dict[str, tuple[int, int]] = {}

    def _slot(self, token: str) -> tuple[int, int]:
        hit = self._cache.get(token)
        if hit is None:
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=8, key=str(self.seed).encode()
            ).digest()
            value = int.from_bytes(digest, "big")
            hit = (value % self.dim, 1 if (value >> 63) & 1 else -1)
            if len(self._cache) < 200_000:
                self._cache[token] = hit
        return hit

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts: dict[str, float] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0.0) + 1.0
            for bigram in _bigrams(tokens):
                counts[bigram] = counts.get(bigram, 0.0) + 0.5
            for token, count in counts.items():
                idx, sign = self._slot(token)
                # sublinear term frequency damps boilerplate repetition
                out[row, idx] += sign * (1.0 + math.log(count))
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class CachingEmbedder(Embedder):
    """Wraps any embedder with an exact-text LRU-ish cache.

    Query embedding is on the hot path and questions repeat heavily in eval
    loops; caching removes that cost without changing results.
    """

    def __init__(self, inner: Embedder, max_entries: int = 20_000):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.max_entries = max_entries
        # keyed by (kind, text) so a string is not served from the query cache
        # when it is wanted as a document, or the reverse
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def embed_query(self, text: str) -> np.ndarray:
        cached = self._cache.get(("q", text))
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vector = self.inner.embed_query(text)
        if len(self._cache) < self.max_entries:
            self._cache[("q", text)] = vector
        return vector

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        pending: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(("d", text))
            if cached is None:
                pending.append(i)
                self.misses += 1
            else:
                out[i] = cached
                self.hits += 1
        if pending:
            fresh = self.inner.embed([texts[i] for i in pending])
            for slot, i in enumerate(pending):
                out[i] = fresh[slot]
                if len(self._cache) < self.max_entries:
                    self._cache[("d", texts[i])] = fresh[slot]
        return out


def pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


class RemoteEmbedder(Embedder):
    """Shared batching, retry and normalisation for hosted or served models.

    Subclasses implement `_embed_batch`. Everything that makes a remote
    embedder usable at corpus scale lives here: batching (an order of magnitude
    faster than one call per chunk), bounded retry, and L2 normalisation on
    ingest so the retrieval layer's `matrix @ query` really is cosine
    similarity rather than an unnormalised dot product.
    """

    #: models returning already-normalised vectors can skip the renormalisation
    normalizes_output = False
    #: prepended to queries only, for instruction-tuned models
    query_prefix = ""

    def __init__(self, model: str, dim: int, *, batch_size: int = 64, retries: int = 3,
                 query_prefix: str | None = None):
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.retries = retries
        if query_prefix is not None:
            self.query_prefix = query_prefix
        self.name = f"{self.provider}:{model}"

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([f"{self.query_prefix}{text}" if self.query_prefix else text])[0]

    provider = "remote"

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            vectors = self._with_retry(chunk)
            for offset, vector in enumerate(vectors):
                row = np.asarray(vector, dtype=np.float32)
                if row.shape[0] != self.dim:
                    raise ValueError(
                        f"{self.name} returned dimension {row.shape[0]}, expected {self.dim}"
                    )
                out[start + offset] = row
        if not self.normalizes_output:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            np.divide(out, norms, out=out, where=norms > 0)
        return out

    def _with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        import time

        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._embed_batch(texts)
            except Exception as exc:
                last = exc
                backoff = 0.5 * (2 ** attempt)
                log.warning("embedding batch failed; retrying", extra={
                    "model": self.name, "attempt": attempt + 1,
                    "backoff_s": backoff, "error": str(exc)[:200],
                })
                time.sleep(backoff)
        raise RuntimeError(f"{self.name}: embedding failed after {self.retries} attempts") from last


#: Query instructions for instruction-tuned local models, keyed by model
#: prefix. The document side is always embedded raw.
QUERY_INSTRUCTIONS: dict[str, str] = {
    "qwen3-embedding": (
        "Instruct: Given a financial question, retrieve relevant passages from "
        "SEC filings\nQuery: "
    ),
    "e5-": "query: ",
    "multilingual-e5": "query: ",
    "bge-": "Represent this sentence for searching relevant passages: ",
}


class OllamaEmbedder(RemoteEmbedder):
    """A locally served embedding model. No key, no token cost, no data egress."""

    provider = "ollama"

    def __init__(self, model: str = "qwen3-embedding:0.6b", dim: int = 1024,
                 *, base_url: str = "http://localhost:11434", **kwargs: Any):
        self.base_url = base_url.rstrip("/")
        if kwargs.get("query_prefix") is None:
            kwargs["query_prefix"] = next(
                (prefix for key, prefix in QUERY_INSTRUCTIONS.items() if key in model), ""
            )
        super().__init__(model, dim, **kwargs)

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": list(texts)},
            timeout=300.0,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings")
        if vectors is None and payload.get("embedding") is not None:
            vectors = [payload["embedding"]]
        if not vectors:
            raise RuntimeError(f"no embeddings returned: {str(payload)[:200]}")
        return vectors


class OpenAIEmbedder(RemoteEmbedder):
    provider = "openai"
    normalizes_output = True  # OpenAI returns unit-norm vectors

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536,
                 *, api_key: str | None = None, base_url: str | None = None, **kwargs: Any):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("`pip install openai` to use OpenAI embeddings") from exc
        import os

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url
        )
        super().__init__(model, dim, **kwargs)

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model, input=list(texts), dimensions=self.dim
        )
        return [item.embedding for item in response.data]


class GoogleEmbedder(RemoteEmbedder):
    provider = "google"

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 3072,
                 *, api_key: str | None = None, **kwargs: Any):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("`pip install google-genai` to use Gemini embeddings") from exc
        import os

        self.client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        super().__init__(model, dim, **kwargs)

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model, contents=list(texts),
            config={"output_dimensionality": self.dim},
        )
        return [item.values for item in response.embeddings]


#: provider -> (factory, default model, default dimension)
EMBEDDING_PROVIDERS: dict[str, tuple[Any, str, int]] = {
    "hashed": (lambda model, dim, **kw: HashedEmbedder(dim=dim), "hashed-ngram", 512),
    "ollama": (lambda model, dim, **kw: OllamaEmbedder(model, dim, **kw),
               "qwen3-embedding:0.6b", 1024),
    "openai": (lambda model, dim, **kw: OpenAIEmbedder(model, dim, **kw),
               "text-embedding-3-small", 1536),
    "google": (lambda model, dim, **kw: GoogleEmbedder(model, dim, **kw),
               "gemini-embedding-001", 3072),
}


_EMBEDDER: Embedder | None = None


def build_embedder(
    provider: str | None = None, model: str | None = None, dim: int | None = None, **kwargs: Any
) -> Embedder:
    """Construct an embedder, falling back to the hashed model on failure."""
    from creditlens.config import get_settings

    settings = get_settings()
    provider = (provider or settings.embedding_provider or "hashed").lower()
    if provider not in EMBEDDING_PROVIDERS:
        log.warning("unknown embedding provider; using the hashed model",
                    extra={"provider": provider, "known": sorted(EMBEDDING_PROVIDERS)})
        provider = "hashed"

    factory, default_model, default_dim = EMBEDDING_PROVIDERS[provider]
    model = model or settings.embedding_model or default_model
    dim = dim or settings.embedding_dim or default_dim
    try:
        return factory(model, dim, **kwargs)
    except Exception as exc:
        log.warning("embedding provider unavailable; using the hashed model",
                    extra={"provider": provider, "model": model, "error": str(exc)})
        return HashedEmbedder(dim=EMBEDDING_PROVIDERS["hashed"][2])


def get_embedder(provider: str | None = None, dim: int | None = None) -> Embedder:
    """Process-wide embedder singleton.

    A singleton because every vector in the store must come from the same model:
    two embedding spaces mixed in one index degrade retrieval silently.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = CachingEmbedder(build_embedder(provider, dim=dim))
    return _EMBEDDER


def reset_embedder() -> None:
    global _EMBEDDER
    _EMBEDDER = None


def describe_embedder() -> dict[str, Any]:
    embedder = get_embedder()
    inner = getattr(embedder, "inner", embedder)
    return {
        "name": embedder.name,
        "provider": getattr(inner, "provider", "hashed"),
        "model": getattr(inner, "model", embedder.name),
        "dim": embedder.dim,
    }


def cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of L2-normalized rows against one query vector."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    qnorm = float(np.linalg.norm(query))
    if qnorm == 0:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    return matrix @ (query / qnorm)
