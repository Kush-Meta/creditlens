"""FastAPI application.

Endpoint design follows the layering: the deterministic finance and retrieval
layers are exposed directly (so they can be tested, benchmarked and used
without paying for an LLM call), and `/api/analyze` composes them through the
agent. Anything the agent can do, you can also do yourself over HTTP.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from creditlens import __version__
from creditlens.agent import tools as tool_mod
from creditlens.agent.evidence import EvidenceLedger
from creditlens.agent.llm import available_providers, build_engine, probe_engine
from creditlens.agent.orchestrator import Orchestrator
from creditlens.agent.tools import ToolContext
from creditlens.api.schemas import (
    AnalyzeRequest,
    CompareRequest,
    EvalRequest,
    HealthResponse,
    IngestRequest,
    ProvenanceRequest,
    SearchRequest,
)
from creditlens.config import get_settings
from creditlens.db.models import AnalysisRun, Chunk, Company, EvalRun, Filing, FinancialFact
from creditlens.db.session import get_session, healthcheck, init_db
from creditlens.finance import ratios as ratio_mod
from creditlens.observability import METRICS, NullTrace, configure_logging, get_logger
from creditlens.observability.context import current_run_id, set_run_id
from creditlens.retrieval import hybrid

log = get_logger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    init_db()
    log.info("creditlens api started", extra={
        "version": __version__, "env": settings.env,
        "model": settings.model, "fingerprint": settings.fingerprint(),
    })
    yield
    log.info("creditlens api stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="CreditLens",
        version=__version__,
        description=(
            "Evidence-grounded credit analysis of public companies. "
            "LLM reasoning over deterministic financial computation."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = set_run_id(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            METRICS.inc("creditlens_http_requests_total",
                        path=request.url.path, status="500")
            log.exception("unhandled request error", extra={"path": request.url.path})
            return JSONResponse(
                status_code=500,
                content={"error": "internal_error", "request_id": request_id},
                headers={"x-request-id": request_id},
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            METRICS.observe("creditlens_http_latency_ms", elapsed_ms, path=request.url.path)
            from creditlens.observability.context import reset_run_id
            reset_run_id(token)
        METRICS.inc("creditlens_http_requests_total",
                    path=request.url.path, status=str(response.status_code))
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
        return response

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    # ---------------- meta ----------------
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(session: Session = Depends(get_session)) -> HealthResponse:
        settings = get_settings()
        counts = {
            "companies": session.scalar(select(func.count(Company.id))) or 0,
            "filings": session.scalar(select(func.count(Filing.id))) or 0,
            "chunks": session.scalar(select(func.count(Chunk.id))) or 0,
            "financial_facts": session.scalar(select(func.count(FinancialFact.id))) or 0,
            "synthetic_companies": session.scalar(
                select(func.count(Company.id)).where(Company.is_synthetic.is_(True))
            ) or 0,
        }
        db_ok = healthcheck()
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            version=__version__,
            database=db_ok,
            corpus=counts,
            llm=probe_engine(),
            settings_fingerprint=settings.fingerprint(),
        )

    @app.get("/metrics", response_class=PlainTextResponse, tags=["meta"])
    def prometheus_metrics() -> str:
        return METRICS.prometheus()

    @app.get("/api/metrics", tags=["meta"])
    def json_metrics() -> dict[str, Any]:
        return METRICS.snapshot()

    @app.get("/api/config", tags=["meta"])
    def config() -> dict[str, Any]:
        settings = get_settings()
        return {
            "version": __version__,
            "llm_provider": settings.llm_provider,
            "providers": available_providers(),
            "model": settings.model,
            "effort": settings.effort,
            "thinking": settings.thinking,
            "embedding_provider": settings.embedding_provider,
            "embedding_dim": settings.embedding_dim,
            "retrieval_top_k": settings.retrieval_top_k,
            "dense_weight": settings.dense_weight,
            "rrf_k": settings.rrf_k,
            "numeric_tolerance_pct": settings.numeric_tolerance_pct,
            "max_tool_iterations": settings.max_tool_iterations,
            "settings_fingerprint": settings.fingerprint(),
            "ratio_catalog": ratio_mod.catalog(),
        }

    # ---------------- corpus ----------------
    @app.get("/api/providers", tags=["meta"])
    def providers() -> dict[str, Any]:
        """Every supported LLM provider and whether its credentials resolve here."""
        return {
            "configured": get_settings().llm_provider,
            "active": probe_engine(),
            "providers": available_providers(),
        }

    @app.get("/api/companies", tags=["corpus"])
    def companies(session: Session = Depends(get_session)) -> dict[str, Any]:
        return tool_mod.tool_list_companies(_ctx(session))

    @app.get("/api/companies/{ticker}/financials", tags=["corpus"])
    def financials(
        ticker: str,
        freq: str = Query("quarterly", pattern="^(quarterly|annual|all)$"),
        periods: int = Query(6, ge=1, le=20),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        return _run_tool(tool_mod.tool_get_financials, session,
                         ticker=ticker, freq=freq, periods=periods)

    @app.get("/api/companies/{ticker}/ratios", tags=["corpus"])
    def ratios(
        ticker: str,
        period: str | None = None,
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        return _run_tool(tool_mod.tool_compute_ratios, session,
                         ticker=ticker, period=period)

    @app.get("/api/companies/{ticker}/trend", tags=["corpus"])
    def trend(
        ticker: str,
        metric: str = Query(..., description="Ratio name or financial concept."),
        freq: str = Query("quarterly", pattern="^(quarterly|annual)$"),
        lookback: int = Query(8, ge=2, le=24),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        return _run_tool(tool_mod.tool_metric_trend, session,
                         ticker=ticker, metric=metric, freq=freq, lookback=lookback)

    @app.get("/api/companies/{ticker}/scorecard", tags=["corpus"])
    def scorecard(
        ticker: str,
        period: str | None = None,
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        return _run_tool(tool_mod.tool_credit_scorecard, session,
                         ticker=ticker, period=period)

    @app.get("/api/companies/{ticker}/coverage", tags=["corpus"])
    def coverage(ticker: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        return _run_tool(tool_mod.tool_data_coverage, session, ticker=ticker)

    @app.post("/api/compare", tags=["corpus"])
    def compare(
        payload: CompareRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        return _run_tool(tool_mod.tool_compare_companies, session,
                         tickers=payload.tickers, period=payload.period,
                         ratios=payload.ratios)

    # ---------------- retrieval ----------------
    @app.post("/api/search", tags=["retrieval"])
    def search(
        payload: SearchRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        result = hybrid.search(
            session, payload.query,
            tickers=payload.tickers, form_types=payload.form_types,
            item_codes=payload.items, fiscal_years=payload.fiscal_years,
            top_k=payload.top_k,
        )
        return result.to_dict(include_text=payload.include_text)

    # ---------------- provenance ----------------
    @app.post("/api/provenance", tags=["provenance"])
    def provenance(
        payload: ProvenanceRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Answer a question about data lineage.

        Deterministic: the reply is read from recorded provenance, not generated,
        so this endpoint cannot invent a source for a number.
        """
        from creditlens.agent.provenance import answer as provenance_answer

        return provenance_answer(session, payload.question).to_dict()

    @app.get("/api/provenance/metric/{ticker}/{metric}", tags=["provenance"])
    def metric_lineage(
        ticker: str,
        metric: str,
        period: str | None = None,
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        """Full lineage tree for one metric, down to the filings behind it."""
        from creditlens.agent.provenance import LineageResolver

        if metric not in ratio_mod.REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"unknown ratio '{metric}'; see /api/config for the catalog",
            )
        resolver = LineageResolver(session, ticker)
        if resolver.company is None:
            raise HTTPException(status_code=404, detail=f"unknown ticker '{ticker}'")
        return resolver.resolve_ratio(metric, period).to_dict()

    # ---------------- analysis ----------------
    @app.post("/api/analyze", tags=["analysis"])
    def analyze(
        payload: AnalyzeRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        orchestrator = Orchestrator(session)
        result = orchestrator.analyze(
            payload.question, tickers=payload.tickers, run_id=current_run_id()
        )
        return result.to_dict(include_trace=payload.include_trace)

    @app.get("/api/runs", tags=["analysis"])
    def runs(
        limit: int = Query(25, ge=1, le=200), session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        rows = session.scalars(
            select(AnalysisRun).order_by(desc(AnalysisRun.created_at)).limit(limit)
        )
        return {"runs": [{
            "run_id": r.id,
            "created_at": r.created_at.isoformat(),
            "question": r.question,
            "tickers": r.tickers,
            "status": r.status,
            "engine": r.engine,
            "model": r.model,
            "latency_ms": round(r.latency_ms, 1),
            "cost_usd": round(r.cost_usd, 6),
            "tool_calls": r.tool_calls,
            "numeric_accuracy": (r.verification or {}).get("numeric_accuracy"),
            "unsupported_claim_rate": (r.verification or {}).get("unsupported_claim_rate"),
            "confidence": (r.verification or {}).get("confidence"),
        } for r in rows]}

    @app.get("/api/runs/{run_id}", tags=["analysis"])
    def run_detail(run_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        row = session.get(AnalysisRun, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
        return {**(row.payload or {}), "trace": row.trace}

    # ---------------- ingestion ----------------
    @app.post("/api/ingest", tags=["ingestion"])
    def ingest(
        payload: IngestRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        from creditlens.ingest.pipeline import ingest_ticker

        try:
            report = ingest_ticker(
                session, payload.ticker,
                max_filings=payload.max_filings,
                forms=tuple(payload.forms),
                min_fiscal_year=payload.min_fiscal_year,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("ingest failed", extra={"ticker": payload.ticker})
            raise HTTPException(status_code=502, detail=f"ingestion failed: {exc}") from exc
        return report.to_dict()

    @app.post("/api/ingest/fixtures", tags=["ingestion"])
    def ingest_fixtures(session: Session = Depends(get_session)) -> dict[str, Any]:
        from creditlens.ingest.fixtures import load_fixtures

        reports = load_fixtures(session)
        return {"loaded": [r.to_dict() for r in reports]}

    @app.delete("/api/companies/{ticker}", tags=["ingestion"])
    def delete_company(ticker: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        from creditlens.ingest.pipeline import delete_company as remove

        if not remove(session, ticker):
            raise HTTPException(status_code=404, detail=f"unknown ticker '{ticker}'")
        return {"deleted": ticker.upper()}

    # ---------------- evaluation ----------------
    @app.post("/api/eval/run", tags=["evaluation"])
    def eval_run(
        payload: EvalRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        from creditlens.eval.runner import run_suite

        engine = build_engine("offline" if payload.engine == "offline" else None)
        result = run_suite(session, suite=payload.suite, limit=payload.limit, engine=engine)
        return result.to_dict()

    @app.get("/api/eval/runs", tags=["evaluation"])
    def eval_runs(
        limit: int = Query(10, ge=1, le=50), session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        rows = session.scalars(
            select(EvalRun).order_by(desc(EvalRun.created_at)).limit(limit)
        )
        return {"runs": [{
            "id": r.id, "created_at": r.created_at.isoformat(), "suite": r.suite,
            "engine": r.engine, "model": r.model, "n_cases": r.n_cases,
            "metrics": r.metrics, "cost_usd": round(r.cost_usd, 6),
            "duration_s": round(r.duration_s, 2),
            "settings_fingerprint": r.settings_fingerprint,
        } for r in rows]}

    @app.get("/api/eval/runs/{eval_id}", tags=["evaluation"])
    def eval_detail(eval_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        row = session.get(EvalRun, eval_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"eval run '{eval_id}' not found")
        return {
            "id": row.id, "suite": row.suite, "engine": row.engine, "model": row.model,
            "created_at": row.created_at.isoformat(), "metrics": row.metrics,
            "results": row.results, "cost_usd": row.cost_usd, "duration_s": row.duration_s,
        }

    # ---------------- frontend ----------------
    # `no-cache` means "revalidate before reuse", not "never cache": the browser
    # still gets a 304 when nothing changed. Without it a cached app.js runs
    # against freshly served HTML, which fails in a way that looks like a code
    # bug rather than a caching one.
    _STATIC_HEADERS = {"Cache-Control": "no-cache"}

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        """Serve the UI with content-hashed asset URLs.

        `Cache-Control` alone does not rescue a browser that already stored the
        old script: it keeps serving it until that entry expires, so new HTML
        runs against stale JavaScript. Stamping the asset URLs with a hash of
        their contents makes every change a new URL, which no cache can get
        wrong.
        """
        path = WEB_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend not built")
        html = path.read_text(encoding="utf-8")
        for asset in ("app.js", "styles.css"):
            html = html.replace(f"/{asset}", f"/{asset}?v={_asset_version(asset)}")
        return HTMLResponse(html, headers=_STATIC_HEADERS)

    @app.get("/app.js", include_in_schema=False)
    def app_js() -> FileResponse:
        return FileResponse(
            WEB_DIR / "app.js", media_type="application/javascript",
            headers=_STATIC_HEADERS,
        )

    @app.get("/styles.css", include_in_schema=False)
    def app_css() -> FileResponse:
        return FileResponse(
            WEB_DIR / "styles.css", media_type="text/css", headers=_STATIC_HEADERS,
        )


def _asset_version(name: str) -> str:
    """Short content hash of a static asset, used to bust browser caches."""
    import hashlib

    path = WEB_DIR / name
    if not path.exists():
        return __version__
    return hashlib.blake2b(path.read_bytes(), digest_size=6).hexdigest()


def _ctx(session: Session) -> ToolContext:
    return ToolContext(session=session, ledger=EvidenceLedger(), trace=NullTrace())


def _run_tool(fn, session: Session, **kwargs) -> dict[str, Any]:
    """Expose a tool as an HTTP endpoint, mapping tool errors to 4xx."""
    try:
        return fn(_ctx(session), **kwargs)
    except tool_mod.ToolError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


app = create_app()
