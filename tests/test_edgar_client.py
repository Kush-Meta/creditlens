"""EDGAR client plumbing: rate limiting, caching, retries, endpoint shapes.

Exercised against a stub transport - the suite must never touch the network.
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

from creditlens.ingest.edgar_client import EdgarClient, FilingRef, RateLimiter


class TestRateLimiter:
    def test_allows_the_burst_immediately(self):
        limiter = RateLimiter(rate_per_s=100, burst=5)
        started = time.monotonic()
        for _ in range(5):
            limiter.acquire()
        assert time.monotonic() - started < 0.05

    def test_throttles_beyond_the_burst(self):
        limiter = RateLimiter(rate_per_s=50, burst=1)
        limiter.acquire()
        started = time.monotonic()
        limiter.acquire()
        assert time.monotonic() - started >= 0.01


class StubTransport(httpx.BaseTransport):
    """Records requests and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        status, body = self.responses.pop(0) if self.responses else (200, "{}")
        return httpx.Response(status, text=body, request=request)


@pytest.fixture
def client_factory(tmp_path):
    def build(responses):
        client = EdgarClient(cache_dir=tmp_path / "cache", rate_limit=1000)
        transport = StubTransport(responses)
        client._client = httpx.Client(transport=transport, timeout=5)
        return client, transport

    return build


class TestFetching:
    def test_responses_are_cached_on_disk(self, client_factory):
        client, transport = client_factory([(200, '{"ok": true}')])
        first = client.fetch_json("https://example.test/a.json")
        second = client.fetch_json("https://example.test/a.json")
        assert first == second == {"ok": True}
        assert len(transport.requests) == 1  # second call served from cache

    def test_retries_transient_failures(self, client_factory):
        client, transport = client_factory([
            (503, "unavailable"), (429, "slow down"), (200, '{"ok": 1}'),
        ])
        assert client.fetch_json("https://example.test/b.json") == {"ok": 1}
        assert len(transport.requests) == 3

    def test_gives_up_after_the_retry_budget(self, client_factory):
        client, _ = client_factory([(500, "boom")] * 3)
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            client.fetch("https://example.test/c.json", retries=3)

    def test_client_errors_are_not_retried(self, client_factory):
        client, transport = client_factory([(404, "missing")])
        with pytest.raises(RuntimeError):
            client.fetch("https://example.test/d.json", retries=1)
        assert len(transport.requests) == 1


class TestEndpoints:
    def test_resolve_cik_pads_to_ten_digits(self, client_factory):
        body = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
        client, _ = client_factory([(200, body)])
        cik, name = client.resolve_cik("aapl")
        assert cik == "0000320193"
        assert name == "Apple Inc."

    def test_unknown_ticker_raises_keyerror(self, client_factory):
        client, _ = client_factory([(200, json.dumps({}))])
        with pytest.raises(KeyError, match="not found"):
            client.resolve_cik("NOSUCH")

    def test_recent_filings_filters_by_form(self, client_factory):
        body = json.dumps({"filings": {"recent": {
            "accessionNumber": ["0000-24-1", "0000-24-2", "0000-24-3"],
            "form": ["10-K", "8-K", "10-Q"],
            "filingDate": ["2024-02-01", "2024-03-01", "2024-05-01"],
            "reportDate": ["2023-12-31", "", "2024-03-31"],
            "primaryDocument": ["k.htm", "e.htm", "q.htm"],
        }}})
        client, _ = client_factory([(200, body)])
        refs = client.recent_filings("0000000123", forms=("10-K", "10-Q"))
        assert [r.form_type for r in refs] == ["10-K", "10-Q"]

    def test_recent_filings_honours_the_limit(self, client_factory):
        body = json.dumps({"filings": {"recent": {
            "accessionNumber": [f"a{i}" for i in range(5)],
            "form": ["10-Q"] * 5,
            "filingDate": ["2024-01-01"] * 5,
            "reportDate": ["2023-12-31"] * 5,
            "primaryDocument": ["q.htm"] * 5,
        }}})
        client, _ = client_factory([(200, body)])
        assert len(client.recent_filings("123", forms=("10-Q",), limit=2)) == 2

    def test_document_url_construction(self):
        ref = FilingRef(
            accession="0000037996-26-000015", form_type="10-K",
            filing_date="2026-02-11", report_date="2025-12-31",
            primary_document="f-20251231.htm", cik="0000037996",
        )
        url = ref.document_url("https://www.sec.gov")
        assert url == (
            "https://www.sec.gov/Archives/edgar/data/37996/"
            "000003799626000015/f-20251231.htm"
        )

    def test_user_agent_is_sent(self, client_factory):
        client, _ = client_factory([(200, "{}")])
        assert "CreditLens" in client.user_agent
