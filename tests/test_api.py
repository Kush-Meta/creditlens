"""HTTP surface."""
from __future__ import annotations

import pytest


class TestMeta:
    def test_health_reports_corpus_and_engine(self, client):
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["database"] is True
        assert payload["corpus"]["companies"] >= 4
        assert "reachable" in payload["llm"]
        assert payload["settings_fingerprint"]

    def test_prometheus_exposition(self, client):
        client.get("/api/companies")
        body = client.get("/metrics").text
        assert "# TYPE" in body
        assert "creditlens_http_requests_total" in body

    def test_config_exposes_the_ratio_catalog(self, client):
        payload = client.get("/api/config").json()
        assert len(payload["ratio_catalog"]) > 20
        assert payload["settings_fingerprint"]

    def test_request_id_is_echoed(self, client):
        response = client.get("/health", headers={"x-request-id": "abc123"})
        assert response.headers["x-request-id"] == "abc123"
        assert "x-response-time-ms" in response.headers


class TestCorpusEndpoints:
    def test_companies(self, client):
        payload = client.get("/api/companies").json()
        assert {c["ticker"] for c in payload["companies"]} >= {"NVCR", "ARMT", "KSTR", "HRBG"}

    def test_financials(self, client):
        payload = client.get("/api/companies/NVCR/financials?periods=4").json()
        assert payload["ticker"] == "NVCR"
        assert len(payload["periods"]) >= 4

    def test_ratios(self, client):
        payload = client.get("/api/companies/NVCR/ratios?period=TTM").json()
        assert payload["period"].startswith("TTM")
        assert len(payload["ratios"]) > 20

    def test_trend(self, client):
        payload = client.get(
            "/api/companies/NVCR/trend?metric=net_debt_to_ebitda&lookback=6"
        ).json()
        assert payload["direction"] == "deteriorating"

    def test_scorecard(self, client):
        payload = client.get("/api/companies/ARMT/scorecard?period=TTM").json()
        assert payload["composite_score_0_100"] > 60
        assert "disclaimer" in payload

    def test_coverage(self, client):
        payload = client.get("/api/companies/NVCR/coverage").json()
        assert "concepts_available" in payload

    def test_unknown_ticker_is_404_with_alternatives(self, client):
        response = client.get("/api/companies/ZZZZ/ratios")
        assert response.status_code == 404
        assert "Available issuers" in response.json()["detail"]

    def test_compare(self, client):
        payload = client.post(
            "/api/compare", json={"tickers": ["ARMT", "HRBG"], "period": "TTM"}
        ).json()
        assert payload["scorecards"]["ranking"][0] == "ARMT"

    def test_compare_requires_two_tickers(self, client):
        assert client.post("/api/compare", json={"tickers": ["ARMT"]}).status_code == 422


class TestSearch:
    def test_search_returns_hits_with_citations(self, client):
        payload = client.post("/api/search", json={
            "query": "borrowing base availability", "top_k": 5,
        }).json()
        assert payload["hits"]
        assert payload["hits"][0]["citation"]["ticker"]

    def test_search_filters(self, client):
        payload = client.post("/api/search", json={
            "query": "risk", "tickers": ["HRBG"], "items": ["1A"], "top_k": 4,
        }).json()
        assert {h["citation"]["ticker"] for h in payload["hits"]} == {"HRBG"}

    def test_search_validates_input(self, client):
        assert client.post("/api/search", json={"query": "x"}).status_code == 422


class TestAnalyze:
    def test_analysis_response_contract(self, client):
        payload = client.post("/api/analyze", json={
            "question": "Has Novacore's credit quality deteriorated?",
            "tickers": ["NVCR"],
        }).json()
        for key in ("run_id", "answer", "credit_direction", "key_metrics",
                    "positive_factors", "risk_factors", "reasoning", "citations",
                    "verification", "tool_calls", "usage", "estimated_cost_usd"):
            assert key in payload, key
        assert payload["credit_direction"] == "deteriorating"
        assert payload["verification"]["citation_validity"] == 1.0

    def test_trace_is_opt_in(self, client):
        body = {"question": "Summarize Aramont's credit profile."}
        assert "trace" not in client.post("/api/analyze", json=body).json()
        body["include_trace"] = True
        assert client.post("/api/analyze", json=body).json()["trace"]

    def test_runs_are_persisted_and_retrievable(self, client):
        run_id = client.post("/api/analyze", json={
            "question": "Summarize Harbridge's credit profile.",
        }).json()["run_id"]
        listing = client.get("/api/runs?limit=5").json()["runs"]
        assert any(r["run_id"] == run_id for r in listing)
        assert client.get(f"/api/runs/{run_id}").json()["run_id"] == run_id

    def test_missing_run_is_404(self, client):
        assert client.get("/api/runs/nope").status_code == 404

    def test_question_is_validated(self, client):
        assert client.post("/api/analyze", json={"question": "?"}).status_code == 422


class TestEvaluation:
    def test_eval_run_returns_metrics(self, client):
        payload = client.post("/api/eval/run", json={
            "suite": "golden", "limit": 3, "engine": "offline",
        }).json()
        headline = payload["metrics"]["headline"]
        assert headline["cases"] >= 1
        assert "numeric_accuracy" in headline
        assert payload["results"]

    def test_eval_history(self, client):
        client.post("/api/eval/run", json={"suite": "golden", "limit": 2, "engine": "offline"})
        runs = client.get("/api/eval/runs").json()["runs"]
        assert runs
        assert runs[0]["engine"] == "offline-deterministic"


class TestFrontend:
    @pytest.mark.parametrize("path", ["/", "/app.js", "/styles.css"])
    def test_static_assets_are_served(self, client, path):
        assert client.get(path).status_code == 200
