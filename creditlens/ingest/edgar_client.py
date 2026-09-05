"""SEC EDGAR client.

EDGAR is a well-behaved but strict host: it requires a descriptive
User-Agent, throttles above ~10 requests/second, and serves large documents.
This client therefore does the three things every production ingestion client
needs and most tutorials skip: a token-bucket rate limiter, retry with
backoff on transient failures, and an on-disk response cache so re-running a
pipeline costs nothing and stays reproducible.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from creditlens.config import get_settings
from creditlens.observability import METRICS, get_logger

log = get_logger(__name__)


class RateLimiter:
    """Simple token bucket, shared across threads."""

    def __init__(self, rate_per_s: float, burst: int | None = None):
        self.rate = rate_per_s
        self.capacity = burst or max(int(rate_per_s), 1)
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                time.sleep(max((1.0 - self._tokens) / self.rate, 0.01))


@dataclass
class FilingRef:
    accession: str
    form_type: str
    filing_date: str
    report_date: str | None
    primary_document: str
    cik: str

    @property
    def accession_nodashes(self) -> str:
        return self.accession.replace("-", "")

    def document_url(self, base: str) -> str:
        return f"{base}/Archives/edgar/data/{int(self.cik)}/{self.accession_nodashes}/{self.primary_document}"


class EdgarClient:
    def __init__(
        self,
        *,
        user_agent: str | None = None,
        cache_dir: Path | None = None,
        rate_limit: float | None = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.user_agent = user_agent or settings.edgar_user_agent
        self.cache_dir = cache_dir or (settings.data_dir / "edgar_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(rate_limit or settings.edgar_rate_limit_per_s)
        self._client = httpx.Client(
            timeout=settings.http_timeout_s,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            follow_redirects=True,
        )

    # -- plumbing ---------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.cache"

    def fetch(self, url: str, *, use_cache: bool = True, retries: int = 3) -> str:
        path = self._cache_path(url)
        if use_cache and path.exists():
            METRICS.inc("creditlens_edgar_requests_total", outcome="cache_hit")
            return path.read_text(encoding="utf-8", errors="replace")

        last_error: Exception | None = None
        for attempt in range(retries):
            self.limiter.acquire()
            try:
                response = self._client.get(url)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request, response=response,
                    )
                response.raise_for_status()
                text = response.text
                path.write_text(text, encoding="utf-8")
                METRICS.inc("creditlens_edgar_requests_total", outcome="ok")
                return text
            except Exception as exc:
                last_error = exc
                backoff = 0.5 * (2 ** attempt)
                log.warning(
                    "edgar fetch failed; retrying",
                    extra={"url": url, "attempt": attempt + 1, "backoff_s": backoff,
                           "error": str(exc)},
                )
                time.sleep(backoff)
        METRICS.inc("creditlens_edgar_requests_total", outcome="error")
        raise RuntimeError(f"EDGAR fetch failed after {retries} attempts: {url}") from last_error

    def fetch_json(self, url: str, *, use_cache: bool = True) -> dict[str, Any]:
        return json.loads(self.fetch(url, use_cache=use_cache))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- endpoints --------------------------------------------------------
    def ticker_map(self) -> dict[str, dict[str, str]]:
        """ticker -> {cik, name}. EDGAR publishes this as a single JSON file."""
        raw = self.fetch_json(f"{self.settings.edgar_www_url}/files/company_tickers.json")
        out: dict[str, dict[str, str]] = {}
        for entry in raw.values():
            ticker = str(entry.get("ticker", "")).upper()
            if ticker:
                out[ticker] = {
                    "cik": str(entry["cik_str"]).zfill(10),
                    "name": entry.get("title", ticker),
                }
        return out

    def resolve_cik(self, ticker: str) -> tuple[str, str]:
        mapping = self.ticker_map()
        entry = mapping.get(ticker.upper())
        if entry is None:
            raise KeyError(f"ticker not found in EDGAR ticker map: {ticker}")
        return entry["cik"], entry["name"]

    def company_facts(self, cik: str) -> dict[str, Any]:
        """XBRL companyfacts: every tagged numeric fact the filer has reported."""
        return self.fetch_json(
            f"{self.settings.edgar_base_url}/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
        )

    def submissions(self, cik: str) -> dict[str, Any]:
        return self.fetch_json(
            f"{self.settings.edgar_base_url}/submissions/CIK{cik.zfill(10)}.json"
        )

    def recent_filings(
        self, cik: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 8
    ) -> list[FilingRef]:
        data = self.submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        rows = zip(
            recent.get("accessionNumber", []),
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("reportDate", []),
            recent.get("primaryDocument", []),
        )
        out: list[FilingRef] = []
        for accession, form, filed, report, primary in rows:
            if form not in forms:
                continue
            out.append(FilingRef(
                accession=accession, form_type=form, filing_date=filed,
                report_date=report or None, primary_document=primary, cik=cik,
            ))
            if len(out) >= limit:
                break
        return out

    def filing_document(self, ref: FilingRef) -> str:
        return self.fetch(ref.document_url(self.settings.edgar_www_url))
