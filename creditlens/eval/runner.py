"""Eval execution.

Two harnesses share the dataset:

* `run_suite` - end-to-end. Runs each question through the real agent loop and
  measures retrieval quality, numeric accuracy against independently recomputed
  ground truth, citation validity, unsupported-claim rate, tool selection,
  latency and cost.
* `run_retrieval_ablation` - retrieval only, no LLM. Compares lexical-only,
  dense-only and fused configurations on the same labelled cases, which is how
  a retrieval change gets justified rather than asserted.

Numeric ground truth is deliberately recomputed here through
`creditlens.finance` directly rather than read from the agent's tool output.
Checking the agent against its own tools would only prove it can copy.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from creditlens.agent.llm import LLMEngine, build_engine
from creditlens.agent.orchestrator import AnalysisResult, Orchestrator
from creditlens.config import get_settings
from creditlens.db.models import EvalRun
from creditlens.eval.dataset import EvalCase, RelevanceSpec, load_suite
from creditlens.eval.metrics import (
    MetricAccumulator,
    compare_metric_values,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    recall_ceiling_normalized,
    reciprocal_rank,
    set_f1,
)
from creditlens.finance import ratios as ratio_mod
from creditlens.finance import statements
from creditlens.observability import get_logger
from creditlens.retrieval import corpus as corpus_mod
from creditlens.retrieval import hybrid

log = get_logger(__name__)


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    sector: str | None = None
    profile: str | None = None
    generated: bool = False
    status: str = "ok"
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tool_calls: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    answer: str = ""
    credit_direction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "sector": self.sector,
            "profile": self.profile,
            "generated": self.generated,
            "question": self.question,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "tool_calls": self.tool_calls,
            "metrics": self.metrics,
            "failures": self.failures,
            "credit_direction": self.credit_direction,
            "answer": self.answer[:600],
        }


@dataclass
class SuiteResult:
    id: str
    suite: str
    engine: str
    model: str
    n_cases: int
    metrics: dict[str, Any]
    results: list[CaseResult]
    cost_usd: float
    duration_s: float
    settings_fingerprint: str
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "engine": self.engine,
            "model": self.model,
            "n_cases": self.n_cases,
            "metrics": self.metrics,
            "cost_usd": round(self.cost_usd, 6),
            "duration_s": round(self.duration_s, 2),
            "settings_fingerprint": self.settings_fingerprint,
            "skipped": self.skipped,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------
#: Resolved label sets are reused across ablation configurations and across
#: the retrieval and citation checks of a single case.
_RELEVANCE_CACHE: dict[tuple, set[int]] = {}


def relevant_chunk_ids(session: Session, spec: RelevanceSpec) -> set[int]:
    """Resolve a relevance predicate against the corpus.

    Metadata narrows the candidate set with the snapshot's integer-coded arrays
    before any substring matching happens. Scanning every chunk's text was fine
    at a few thousand chunks; at seventy thousand, across two hundred cases and
    five ablation configurations, it dominated the entire evaluation run.
    """
    snapshot = corpus_mod.get_snapshot(session)
    key = (
        snapshot.version,
        tuple(sorted(spec.tickers)), tuple(sorted(spec.items)),
        tuple(sorted(spec.form_types)),
        tuple(tuple(sorted(group)) for group in spec.any_of),
    )
    cached = _RELEVANCE_CACHE.get(key)
    if cached is not None:
        return cached

    indices = corpus_mod.filter_indices(
        snapshot,
        tickers=spec.tickers or None,
        item_codes=spec.items or None,
        form_types=spec.form_types or None,
    )
    groups = [[term.lower() for term in group] for group in spec.any_of]
    relevant: set[int] = set()
    for index in indices:
        record = snapshot.records[int(index)]
        text = record.text.lower()
        if all(any(term in text for term in group) for group in groups):
            relevant.add(record.chunk_id)

    _RELEVANCE_CACHE[key] = relevant
    return relevant


def ground_truth_metric(
    session: Session, ticker: str, period: str, ratio: str
) -> float | None:
    """Recompute a metric straight from the finance layer, bypassing the agent."""
    periods = statements.load_periods(session, ticker, freq="quarterly")
    if not periods:
        periods = statements.load_periods(session, ticker, freq="annual")
    target = statements.select_period(periods, period)
    if target is None:
        return None
    if ratio in ratio_mod.REGISTRY:
        return ratio_mod.compute(ratio, target).value
    return target.get(ratio)


# ---------------------------------------------------------------------------
# end-to-end suite
# ---------------------------------------------------------------------------
def run_suite(
    session: Session,
    *,
    suite: str = "golden",
    limit: int | None = None,
    engine: LLMEngine | None = None,
    cases: Sequence[EvalCase] | None = None,
    persist: bool = True,
) -> SuiteResult:
    settings = get_settings()
    engine = engine or build_engine()
    all_cases = list(cases) if cases is not None else load_suite(suite)
    if limit:
        all_cases = all_cases[:limit]

    available = {r.ticker for r in corpus_mod.get_snapshot(session).records}
    runnable, skipped = [], []
    for case in all_cases:
        missing = [t for t in case.tickers if t.upper() not in available]
        if missing:
            skipped.append(f"{case.id}: issuer(s) not ingested: {', '.join(missing)}")
        else:
            runnable.append(case)

    accumulator = MetricAccumulator()
    results: list[CaseResult] = []
    started = time.perf_counter()
    total_cost = 0.0

    for case in runnable:
        result = _run_case(session, case, engine, accumulator)
        results.append(result)
        total_cost += result.cost_usd

    duration = time.perf_counter() - started
    summary = accumulator.summary()
    summary["headline"] = _headline(summary, results, duration)
    summary["slices"] = {
        "by_category": _slice(results, lambda r: r.category),
        "by_sector": _slice(results, lambda r: r.sector),
        "by_profile": _slice(results, lambda r: r.profile),
        "by_origin": _slice(results, lambda r: "generated" if r.generated else "hand-written"),
    }

    suite_result = SuiteResult(
        id=uuid.uuid4().hex,
        suite=suite,
        engine=engine.name,
        model=engine.model,
        n_cases=len(runnable),
        metrics=summary,
        results=results,
        cost_usd=total_cost,
        duration_s=duration,
        settings_fingerprint=settings.fingerprint(),
        skipped=skipped,
    )

    if persist:
        _persist(session, suite_result)
    log.info("eval suite complete", extra={
        "suite": suite, "engine": engine.name, "cases": len(runnable),
        "duration_s": round(duration, 2), "cost_usd": round(total_cost, 6),
        **suite_result.metrics["headline"],
    })
    return suite_result


def _run_case(
    session: Session, case: EvalCase, engine: LLMEngine, acc: MetricAccumulator
) -> CaseResult:
    orchestrator = Orchestrator(session, engine=engine, persist=False)
    started = time.perf_counter()
    try:
        analysis = orchestrator.analyze(case.question, tickers=case.tickers or None)
        status = analysis.status
    except Exception as exc:
        log.exception("eval case crashed", extra={"case": case.id})
        acc.bump("cases_crashed")
        return CaseResult(
            case_id=case.id, category=case.category, question=case.question,
            sector=case.sector, profile=case.profile, generated=case.generated,
            status="crashed", latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=0.0, failures=[f"{type(exc).__name__}: {exc}"],
        )

    metrics: dict[str, Any] = {}
    failures: list[str] = []
    tool_names = [t["tool"] for t in analysis.tool_calls]

    # --- verification-derived metrics -----------------------------------
    verification = analysis.verification
    metrics["numeric_accuracy"] = verification["numeric_accuracy"]
    metrics["unsupported_claim_rate"] = verification["unsupported_claim_rate"]
    metrics["citation_validity"] = verification["citation_validity"]
    metrics["numeric_claims"] = verification["numeric_claims_checked"]
    acc.add("numeric_accuracy", verification["numeric_accuracy"])
    acc.add("unsupported_claim_rate", verification["unsupported_claim_rate"])
    acc.add("citation_validity", verification["citation_validity"])
    if verification["numeric_claims_contradicted"]:
        acc.bump("cases_with_contradicted_figures")
        failures.append(
            f"{verification['numeric_claims_contradicted']} contradicted figure(s)"
        )
    if verification["invalid_citations"]:
        acc.bump("cases_with_invalid_citations")
        failures.append(f"invalid citations: {verification['invalid_citations']}")

    # --- independent numeric ground truth --------------------------------
    if case.expected_metrics:
        checks, passed, period_matches = 0, 0, 0
        for spec in case.expected_metrics:
            wanted_period = spec.get("period", "latest")
            # Establish computability first. Some issuers genuinely cannot
            # support a metric - Ford tags no consolidated debt - and scoring
            # an answer for not reporting an uncomputable figure would punish
            # exactly the honest behaviour the system is built for.
            baseline = ground_truth_metric(
                session, spec["ticker"], wanted_period, spec["ratio"]
            )
            reported, reported_period = _find_reported(analysis, spec)
            if baseline is None and reported is None:
                continue
            if reported is None:
                checks += 1
                failures.append(
                    f"{spec['ticker']} {spec['ratio']}: not reported in key_metrics"
                )
                continue
            # Correctness is judged against the period the answer *claims*, not
            # the period the case happened to name. Reporting TTM instead of
            # FY2026 is a scope difference, tracked separately; reporting a
            # wrong number for the period you named is an error.
            truth = ground_truth_metric(
                session, spec["ticker"], reported_period or wanted_period, spec["ratio"]
            )
            if truth is None:
                truth = baseline
            if truth is None:
                continue
            checks += 1
            if _periods_equal(reported_period, wanted_period):
                period_matches += 1
            ok, deviation = compare_metric_values(reported, truth, tolerance_pct=1.0)
            if ok:
                passed += 1
            else:
                failures.append(
                    f"{spec['ticker']} {spec['ratio']} ({reported_period}): reported "
                    f"{reported!r}, ground truth {truth:.4f}"
                    + (f" ({deviation:.2f}% off)" if deviation is not None else "")
                )
        if checks:
            score = passed / checks
            metrics["ground_truth_metric_accuracy"] = round(score, 4)
            metrics["reported_period_match"] = round(period_matches / checks, 4)
            acc.add("ground_truth_metric_accuracy", score)
            acc.add("reported_period_match", period_matches / checks)

    # --- retrieval --------------------------------------------------------
    if case.relevance:
        retrieval = _score_retrieval(session, case)
        metrics.update(retrieval["metrics"])
        for name, value in retrieval["metrics"].items():
            acc.add(name, value)
        # did the agent's own run actually surface relevant passages?
        cited_ids = {c["chunk_id"] for c in analysis.citations}
        relevant = retrieval["relevant_ids"]
        if relevant:
            grounded = len(cited_ids & relevant) / max(len(cited_ids), 1)
            metrics["cited_passage_relevance"] = round(grounded, 4)
            acc.add("cited_passage_relevance", grounded)

    # --- tool selection ---------------------------------------------------
    if case.expected_tools:
        scores = set_f1(tool_names, case.expected_tools)
        metrics["tool_selection_f1"] = round(scores["f1"], 4)
        acc.add("tool_selection_f1", scores["f1"])
        acc.add("tool_selection_recall", scores["recall"])
        missing_tools = set(case.expected_tools) - set(tool_names)
        if missing_tools:
            failures.append(f"tools not used: {sorted(missing_tools)}")

    # --- direction --------------------------------------------------------
    if case.expected_direction:
        correct = analysis.credit_direction == case.expected_direction
        metrics["direction_correct"] = float(correct)
        acc.add("direction_accuracy", float(correct))
        if not correct:
            failures.append(
                f"direction {analysis.credit_direction!r}, expected {case.expected_direction!r}"
            )

    # --- content constraints ---------------------------------------------
    narrative = " ".join([analysis.answer, analysis.reasoning,
                          *analysis.risk_factors, *analysis.positive_factors,
                          *analysis.caveats]).lower()
    for phrase in case.must_mention:
        if phrase.lower() not in narrative:
            failures.append(f"missing required mention: {phrase!r}")
    for phrase in case.must_not_mention:
        if phrase.lower() in narrative:
            failures.append(f"contains forbidden phrase: {phrase!r}")
    acc.add("content_constraints_passed",
            0.0 if any(f.startswith(("missing required", "contains forbidden"))
                       for f in failures) else 1.0)

    acc.add("latency_ms", analysis.latency_ms)
    acc.add("cost_usd", analysis.cost_usd)
    acc.add("tool_calls", len(tool_names))
    acc.bump("cases_run")
    if status != "ok":
        acc.bump("cases_incomplete")

    return CaseResult(
        case_id=case.id, category=case.category, question=case.question,
        sector=case.sector, profile=case.profile, generated=case.generated,
        status=status, latency_ms=analysis.latency_ms, cost_usd=analysis.cost_usd,
        tool_calls=tool_names, metrics=metrics, failures=failures,
        answer=analysis.answer, credit_direction=analysis.credit_direction,
    )


def _find_reported(
    analysis: AnalysisResult, spec: dict[str, str]
) -> tuple[str | None, str | None]:
    """Locate the value AND the period the answer reported for a metric."""
    ratio_label = (
        ratio_mod.REGISTRY[spec["ratio"]].label.lower()
        if spec["ratio"] in ratio_mod.REGISTRY else spec["ratio"].lower()
    )
    wanted_ticker = spec["ticker"].upper()
    for metric in analysis.key_metrics:
        if metric.get("ticker", "").upper() not in ("", wanted_ticker):
            continue
        name = str(metric.get("name", "")).lower()
        if ratio_label in name or spec["ratio"].replace("_", " ") in name:
            return metric.get("value"), metric.get("period")
    return None, None


def _periods_equal(reported: str | None, expected: str) -> bool:
    """Compare period labels, tolerating the year suffix on rendered labels."""
    if not reported:
        return False
    if expected.lower() in {"latest", "current"}:
        return True

    def normalize(label: str) -> str:
        return label.upper().replace("-", "").replace(" ", "").replace("_", "")

    left, right = normalize(reported), normalize(expected)
    if left == right:
        return True
    # "TTM2026" satisfies a request for "TTM"; "FY2026" does not satisfy "FY2025".
    return right in {"TTM"} and left.startswith(right)


def _score_retrieval(session: Session, case: EvalCase) -> dict[str, Any]:
    settings = get_settings()
    relevant = relevant_chunk_ids(session, case.relevance)
    query = case.retrieval_query or case.question
    result = hybrid.search(
        session, query,
        tickers=case.tickers or None,
        top_k=settings.retrieval_top_k,
    )
    retrieved = [hit.record.chunk_id for hit in result.chunks]
    k = settings.retrieval_top_k
    return {
        "relevant_ids": relevant,
        "metrics": {
            "retrieval_relevant_total": len(relevant),
            "retrieval_recall_at_k": round(recall_at_k(retrieved, relevant, k), 4),
            "retrieval_recall_normalized": round(
                recall_ceiling_normalized(retrieved, relevant, k), 4
            ),
            "retrieval_precision_at_k": round(precision_at_k(retrieved, relevant, k), 4),
            "retrieval_mrr": round(reciprocal_rank(retrieved, relevant), 4),
            "retrieval_ndcg_at_k": round(ndcg_at_k(retrieved, relevant, k), 4),
        },
    }


#: metrics worth reading per slice; latency and cost stay in the headline
SLICE_METRICS = (
    "numeric_accuracy", "unsupported_claim_rate", "citation_validity",
    "ground_truth_metric_accuracy", "retrieval_ndcg_at_k", "tool_selection_f1",
)


def _slice(
    results: list[CaseResult], key: Callable[[CaseResult], str | None]
) -> dict[str, dict[str, Any]]:
    """Aggregate the headline metrics within each value of a slice dimension.

    A single corpus-wide average hides the thing that matters most as the
    universe grows: whether the system is uniformly good, or good on
    technology issuers and poor on utilities.
    """
    buckets: dict[str, list[CaseResult]] = {}
    for result in results:
        name = key(result)
        if name is None:
            continue
        buckets.setdefault(name, []).append(result)

    out: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(buckets.items()):
        entry: dict[str, Any] = {"cases": len(rows)}
        for metric in SLICE_METRICS:
            values = [
                r.metrics[metric] for r in rows
                if isinstance(r.metrics.get(metric), (int, float))
            ]
            entry[metric] = round(sum(values) / len(values), 4) if values else None
        latencies = [r.latency_ms for r in rows if r.latency_ms]
        entry["latency_p50_ms"] = round(percentile(latencies, 0.5), 1) if latencies else None
        out[name] = entry
    return out


def _headline(summary: dict[str, Any], results: list[CaseResult], duration: float) -> dict[str, Any]:
    def mean(name: str) -> float | None:
        entry = summary.get(name)
        return None if not entry or not entry.get("n") else entry["mean"]

    latency = summary.get("latency_ms", {})
    return {
        "cases": len(results),
        "numeric_accuracy": mean("numeric_accuracy"),
        "unsupported_claim_rate": mean("unsupported_claim_rate"),
        "citation_validity": mean("citation_validity"),
        "ground_truth_metric_accuracy": mean("ground_truth_metric_accuracy"),
        "reported_period_match": mean("reported_period_match"),
        "retrieval_recall_at_k": mean("retrieval_recall_at_k"),
        "retrieval_recall_normalized": mean("retrieval_recall_normalized"),
        "retrieval_ndcg_at_k": mean("retrieval_ndcg_at_k"),
        "tool_selection_f1": mean("tool_selection_f1"),
        "direction_accuracy": mean("direction_accuracy"),
        "latency_p50_ms": latency.get("p50"),
        "latency_p95_ms": latency.get("p95"),
        "mean_cost_usd": mean("cost_usd"),
        "total_duration_s": round(duration, 2),
    }


def _persist(session: Session, result: SuiteResult) -> None:
    try:
        session.add(EvalRun(
            id=result.id, suite=result.suite, engine=result.engine, model=result.model,
            settings_fingerprint=result.settings_fingerprint, n_cases=result.n_cases,
            metrics=result.metrics, results=[r.to_dict() for r in result.results],
            cost_usd=result.cost_usd, duration_s=result.duration_s,
        ))
        session.commit()
    except Exception:
        session.rollback()
        log.exception("failed to persist eval run")


# ---------------------------------------------------------------------------
# retrieval ablation
# ---------------------------------------------------------------------------
ABLATIONS: dict[str, dict[str, Any]] = {
    "lexical_only": {"dense_weight": 0.0, "use_mmr": False},
    "dense_only": {"dense_weight": 1.0, "use_mmr": False},
    "hybrid_rrf_50_50": {"dense_weight": 0.5, "use_mmr": False},
    "hybrid_rrf_mmr": {"dense_weight": 0.5, "use_mmr": True},
    # the configuration actually shipped, so the table always shows what runs
    "shipped_default": {"dense_weight": None, "use_mmr": None},
}


def run_retrieval_ablation(
    session: Session, *, suite: str = "golden", cases: Sequence[EvalCase] | None = None
) -> dict[str, Any]:
    """Compare retrieval configurations on the labelled cases. No LLM involved."""
    settings = get_settings()
    all_cases = [
        c for c in (list(cases) if cases is not None else load_suite(suite))
        if c.relevance is not None
    ]
    available = {r.ticker for r in corpus_mod.get_snapshot(session).records}
    all_cases = [
        c for c in all_cases if all(t.upper() in available for t in c.tickers)
    ]

    # Resolve every label set once, before the configuration loop.
    labels = {case.id: relevant_chunk_ids(session, case.relevance) for case in all_cases}
    all_cases = [c for c in all_cases if labels[c.id]]

    original_weight = settings.dense_weight
    table: dict[str, Any] = {}
    try:
        for name, config in ABLATIONS.items():
            settings.dense_weight = (
                original_weight if config["dense_weight"] is None
                else config["dense_weight"]
            )
            acc = MetricAccumulator()
            for case in all_cases:
                relevant = labels[case.id]
                result = hybrid.search(
                    session, case.retrieval_query or case.question,
                    tickers=case.tickers or None,
                    top_k=settings.retrieval_top_k,
                    use_mmr=(
                        settings.retrieval_use_mmr if config["use_mmr"] is None
                        else config["use_mmr"]
                    ),
                )
                retrieved = [hit.record.chunk_id for hit in result.chunks]
                k = settings.retrieval_top_k
                acc.add("recall_at_k", recall_at_k(retrieved, relevant, k))
                acc.add("recall_normalized", recall_ceiling_normalized(retrieved, relevant, k))
                acc.add("precision_at_k", precision_at_k(retrieved, relevant, k))
                acc.add("mrr", reciprocal_rank(retrieved, relevant))
                acc.add("ndcg_at_k", ndcg_at_k(retrieved, relevant, k))
            summary = acc.summary()
            table[name] = {
                metric: summary.get(metric, {}).get("mean")
                for metric in ("recall_normalized", "recall_at_k", "precision_at_k",
                               "mrr", "ndcg_at_k")
            }
    finally:
        settings.dense_weight = original_weight

    return {
        "suite": suite,
        "cases_scored": len(all_cases),
        "top_k": settings.retrieval_top_k,
        "configurations": table,
        "note": (
            "Relevance labels are programmatic (issuer + filing item + required "
            "terms), so these numbers compare configurations against each other "
            "rather than against human judgement."
        ),
    }
