"""The controlled tool loop.

One agent, one loop, a fixed tool surface, hard budgets. No planner/critic
hierarchy, no dynamic agent spawning: for a bounded analytical task those add
latency and failure modes without improving answers, and they make the trace
much harder to reason about.

Control flow:

    build transcript -> [ model turn -> execute tools -> append results ] * N
                   -> forced `submit_analysis` if the budget runs out
                   -> verify -> assemble -> persist

Guarantees the loop enforces regardless of what the model does:

* every tool result is deterministic code output, recorded in the evidence
  ledger before it re-enters the conversation;
* the loop always terminates - iteration cap, tool-call cap, and a forced
  final turn;
* an answer is always produced, even on model failure, by falling back to the
  deterministic engine and saying so;
* the response carries the verification report, so an unverified number is
  visible to the caller rather than buried.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from creditlens.agent import prompts
from creditlens.agent import tools as tool_mod
from creditlens.agent.evidence import EvidenceLedger
from creditlens.agent.llm import (
    AssistantTurn,
    Conversation,
    LLMEngine,
    LLMError,
    LLMResponse,
    OfflineEngine,
    ToolResult,
    ToolResultBatch,
    Usage,
    UserMessage,
    build_engine,
)
from creditlens.agent.tools import TERMINAL_TOOL, TOOL_SCHEMAS, ToolContext
from creditlens.agent.verifier import VerificationReport, verify
from creditlens.config import get_settings
from creditlens.db.models import AnalysisRun
from creditlens.observability import METRICS, Trace, get_logger

log = get_logger(__name__)


@dataclass
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    is_error: bool
    duration_ms: float
    result_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "is_error": self.is_error,
            "duration_ms": round(self.duration_ms, 2),
            "result_preview": self.result_preview,
        }


@dataclass
class AnalysisResult:
    run_id: str
    question: str
    answer: str
    credit_direction: str
    key_metrics: list[dict[str, Any]]
    positive_factors: list[str]
    risk_factors: list[str]
    reasoning: str
    caveats: list[str]
    citations: list[dict[str, Any]]
    verification: dict[str, Any]
    evidence: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    engine: str
    provider: str
    model: str
    usage: dict[str, Any]
    cost_usd: float
    latency_ms: float
    trace: list[dict[str, Any]]
    trace_summary: dict[str, Any]
    settings_fingerprint: str
    status: str = "ok"
    error: str | None = None
    degraded_reason: str | None = None
    contains_synthetic_data: bool = False

    def to_dict(self, include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "question": self.question,
            "answer": self.answer,
            "credit_direction": self.credit_direction,
            "key_metrics": self.key_metrics,
            "positive_factors": self.positive_factors,
            "risk_factors": self.risk_factors,
            "reasoning": self.reasoning,
            "caveats": self.caveats,
            "citations": self.citations,
            "verification": self.verification,
            "evidence": self.evidence,
            "tool_calls": self.tool_calls,
            "engine": self.engine,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage,
            "estimated_cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "settings_fingerprint": self.settings_fingerprint,
            "status": self.status,
            "error": self.error,
            "degraded_reason": self.degraded_reason,
            "contains_synthetic_data": self.contains_synthetic_data,
            "trace_summary": self.trace_summary,
        }
        if include_trace:
            payload["trace"] = self.trace
        return payload


class Orchestrator:
    def __init__(
        self,
        session: Session,
        *,
        engine: LLMEngine | None = None,
        persist: bool = True,
    ):
        self.session = session
        self.settings = get_settings()
        self.engine = engine or build_engine()
        self.persist = persist

    # -- public ----------------------------------------------------------
    def analyze(
        self,
        question: str,
        *,
        tickers: Sequence[str] | None = None,
        run_id: str | None = None,
    ) -> AnalysisResult:
        run_id = run_id or uuid.uuid4().hex
        started = dt.datetime.now(dt.UTC)
        trace = Trace(run_id)
        ledger = EvidenceLedger()
        ctx = ToolContext(session=self.session, ledger=ledger, trace=trace)
        usage = Usage()
        invocations: list[ToolInvocation] = []
        degraded_reason: str | None = None
        engine = self.engine

        with trace, trace.span("analysis", kind="server", question=question[:200]) as root:
            conversation = self._build_conversation(ctx, question, tickers)
            final_payload: dict[str, Any] | None = None
            status = "ok"
            error: str | None = None

            for iteration in range(self.settings.max_tool_iterations):
                force_final = (
                    iteration == self.settings.max_tool_iterations - 1
                    or len(invocations) >= self.settings.max_tool_calls
                )
                try:
                    with trace.span(
                        "llm.turn", kind="client",
                        iteration=iteration, engine=engine.name, forced=force_final,
                    ) as span:
                        response = engine.complete(
                            system=prompts.SYSTEM_PROMPT,
                            conversation=conversation,
                            tools=TOOL_SCHEMAS,
                            force_tool=TERMINAL_TOOL if force_final else None,
                        )
                        usage.add(response.usage)
                        span.set(
                            provider=engine.provider,
                            stop_reason=response.stop_reason,
                            tool_calls=[c.name for c in response.tool_calls],
                            output_tokens=response.usage.output_tokens,
                            cache_read_tokens=response.usage.cache_read_tokens,
                        )
                except LLMError as exc:
                    log.warning("llm turn failed", extra={"error": str(exc),
                                                          "iteration": iteration})
                    if isinstance(engine, OfflineEngine):
                        status, error = "error", str(exc)
                        break
                    degraded_reason = (
                        f"{engine.provider or engine.name} unavailable ({exc}); "
                        f"answered with the deterministic engine"
                    )
                    engine = OfflineEngine()
                    continue

                if not response.wants_tools:
                    # Model answered in prose without the terminal tool.
                    final_payload = self._payload_from_text(response, ledger)
                    break

                conversation.append(AssistantTurn(response))

                terminal = next(
                    (c for c in response.tool_calls if c.name == TERMINAL_TOOL), None
                )
                if terminal is not None:
                    final_payload = dict(terminal.arguments)
                    break

                results: list[ToolResult] = []
                for call in response.tool_calls:
                    if len(invocations) >= self.settings.max_tool_calls:
                        results.append(self._tool_result(
                            call,
                            {"error": "tool call budget exhausted; submit your "
                                      "analysis with the evidence gathered so far"},
                            is_error=True,
                        ))
                        continue
                    before = dt.datetime.now(dt.UTC)
                    result, is_error = tool_mod.execute(ctx, call.name, call.arguments)
                    elapsed = (dt.datetime.now(dt.UTC) - before).total_seconds() * 1000
                    invocations.append(ToolInvocation(
                        name=call.name, arguments=call.arguments, is_error=is_error,
                        duration_ms=elapsed,
                        result_preview=_preview(result),
                    ))
                    results.append(self._tool_result(call, result, is_error))
                conversation.append(ToolResultBatch(results))

            if final_payload is None and status == "ok":
                status = "incomplete"
                error = "the agent did not produce a final analysis within its budget"
                final_payload = self._fallback_payload(ledger)

            report = verify(
                narrative=_narrative_of(final_payload or {}),
                ledger=ledger,
                stated_confidence=(final_payload or {}).get("confidence"),
                declared_citations=(final_payload or {}).get("citations"),
            )
            root.set(
                status=status,
                tool_calls=len(invocations),
                numeric_claims=len(report.numeric_claims),
                confidence=report.confidence,
            )

        latency_ms = (dt.datetime.now(dt.UTC) - started).total_seconds() * 1000
        result = self._assemble(
            run_id=run_id, question=question, payload=final_payload or {},
            ledger=ledger, report=report, invocations=invocations, usage=usage,
            trace=trace, latency_ms=latency_ms, engine=engine, status=status,
            error=error, degraded_reason=degraded_reason,
        )
        if self.persist:
            self._persist(result)

        METRICS.inc("creditlens_analysis_total", outcome=result.status)
        METRICS.observe("creditlens_analysis_latency_ms", latency_ms)
        METRICS.observe("creditlens_analysis_cost_usd", result.cost_usd)
        log.info("analysis complete", extra={
            "run_id": run_id, "status": result.status, "engine": result.engine,
            "tool_calls": len(invocations), "latency_ms": round(latency_ms, 1),
            "cost_usd": round(result.cost_usd, 6),
            "numeric_accuracy": report.numeric_accuracy,
            "unsupported_rate": report.unsupported_rate,
        })
        return result

    # -- internals -------------------------------------------------------
    def _build_conversation(
        self, ctx: ToolContext, question: str, tickers: Sequence[str] | None
    ) -> Conversation:
        listing = tool_mod.tool_list_companies(ctx)
        companies = listing["companies"]
        preamble = prompts.corpus_preamble(
            companies, synthetic_warning=any(c["is_synthetic"] for c in companies)
        )
        scope = (
            f"\n\nThe user scoped this question to: {', '.join(t.upper() for t in tickers)}."
            if tickers else ""
        )
        return [UserMessage(f"{preamble}{scope}\n\nQUESTION: {question}")]

    def _tool_result(
        self, call: Any, result: dict[str, Any], is_error: bool
    ) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=json.dumps(result, default=str),
            is_error=is_error,
        )

    def _payload_from_text(self, response: LLMResponse, ledger: EvidenceLedger) -> dict[str, Any]:
        return {
            "answer": response.text or "The agent produced no answer.",
            "credit_direction": "not_applicable",
            "key_metrics": [],
            "positive_factors": [],
            "risk_factors": [],
            "reasoning": response.text,
            "citations": sorted(ledger.known_labels()),
            "confidence": "low",
            "caveats": [
                "The agent answered in prose without submitting the structured "
                "analysis; fields other than the narrative are unpopulated."
            ],
        }

    def _fallback_payload(self, ledger: EvidenceLedger) -> dict[str, Any]:
        summary = ledger.summary()
        return {
            "answer": (
                "The analysis did not complete within its tool budget. "
                f"Evidence gathered: {summary['numeric_facts']} computed figures and "
                f"{summary['passages']} filing passages."
            ),
            "credit_direction": "not_applicable",
            "key_metrics": [],
            "positive_factors": [],
            "risk_factors": [],
            "reasoning": "",
            "citations": sorted(ledger.known_labels()),
            "confidence": "low",
            "caveats": ["Incomplete run: the iteration or tool-call budget was exhausted."],
        }

    def _assemble(
        self, *, run_id: str, question: str, payload: dict[str, Any],
        ledger: EvidenceLedger, report: VerificationReport,
        invocations: list[ToolInvocation], usage: Usage, trace: Trace,
        latency_ms: float, engine: LLMEngine, status: str, error: str | None,
        degraded_reason: str | None,
    ) -> AnalysisResult:
        used_labels = set(report.valid_citations) or set(ledger.known_labels())
        return AnalysisResult(
            run_id=run_id,
            question=question,
            answer=str(payload.get("answer", "")),
            credit_direction=str(payload.get("credit_direction", "not_applicable")),
            key_metrics=list(payload.get("key_metrics") or []),
            positive_factors=list(payload.get("positive_factors") or []),
            risk_factors=list(payload.get("risk_factors") or []),
            reasoning=str(payload.get("reasoning", "")),
            caveats=list(payload.get("caveats") or []),
            citations=ledger.citations(used_labels),
            verification=report.to_dict(),
            evidence={
                **ledger.summary(),
                "numeric_evidence": [n.to_dict() for n in ledger.numeric_values()],
            },
            tool_calls=[i.to_dict() for i in invocations],
            engine=engine.name,
            provider=engine.provider,
            model=engine.model,
            usage=usage.to_dict(engine.model, engine.provider),
            cost_usd=usage.cost_usd(engine.model, engine.provider),
            latency_ms=latency_ms,
            trace=trace.to_list(),
            trace_summary=trace.summary(),
            settings_fingerprint=self.settings.fingerprint(),
            status=status,
            error=error,
            degraded_reason=degraded_reason,
            contains_synthetic_data=ledger.has_synthetic(),
        )

    def _persist(self, result: AnalysisResult) -> None:
        try:
            self.session.merge(AnalysisRun(
                id=result.run_id,
                question=result.question,
                tickers=",".join(sorted({
                    m.get("ticker", "") for m in result.key_metrics if m.get("ticker")
                })) or None,
                answer=result.answer,
                payload=result.to_dict(include_trace=False),
                verification=result.verification,
                trace=result.trace,
                model=result.model,
                engine=result.engine,
                settings_fingerprint=result.settings_fingerprint,
                latency_ms=result.latency_ms,
                input_tokens=result.usage.get("input_tokens", 0),
                output_tokens=result.usage.get("output_tokens", 0),
                cached_tokens=result.usage.get("cache_read_tokens", 0),
                cost_usd=result.cost_usd,
                tool_calls=len(result.tool_calls),
                status=result.status,
                error=result.error,
            ))
            self.session.commit()
        except Exception:
            self.session.rollback()
            log.exception("failed to persist analysis run", extra={"run_id": result.run_id})


def _narrative_of(payload: dict[str, Any]) -> str:
    """The text the verifier checks: everything the answer asserts in prose."""
    parts = [
        str(payload.get("answer", "")),
        str(payload.get("reasoning", "")),
        *[str(x) for x in payload.get("positive_factors") or []],
        *[str(x) for x in payload.get("risk_factors") or []],
        *[str(x) for x in payload.get("caveats") or []],
    ]
    for metric in payload.get("key_metrics") or []:
        parts.append(
            f"{metric.get('name', '')} {metric.get('value', '')} "
            f"{metric.get('commentary', '')}"
        )
    return "\n".join(p for p in parts if p)


def _preview(result: dict[str, Any], limit: int = 400) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + "..."
