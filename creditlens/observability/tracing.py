"""Minimal in-process tracer.

Why not OpenTelemetry: the deliverable needs zero-dependency, inspectable
traces that ship with the API response ("show me exactly which tools ran, in
what order, how long each took, and what they cost"). The span model below is
OTel-shaped on purpose - `to_otlp_like()` emits a structure that maps 1:1 onto
an OTLP span, so swapping in a real exporter is a ~30 line change.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from creditlens.observability import context as ctx
from creditlens.observability.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    run_id: str | None
    kind: str = "internal"
    start_ns: int = field(default_factory=time.perf_counter_ns)
    end_ns: int | None = None
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.perf_counter_ns()
        return (end - self.start_ns) / 1e6

    def set(self, **attrs: Any) -> Span:
        self.attributes.update(attrs)
        return self

    def event(self, name: str, **attrs: Any) -> Span:
        self.events.append({"name": name, "ts_ms": self.duration_ms, **attrs})
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "events": self.events,
        }

    def to_otlp_like(self) -> dict[str, Any]:
        return {
            "traceId": self.run_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "startTimeUnixNano": self.start_ns,
            "endTimeUnixNano": self.end_ns,
            "status": {"code": "STATUS_CODE_OK" if self.status == "ok" else "STATUS_CODE_ERROR"},
            "attributes": [{"key": k, "value": v} for k, v in self.attributes.items()],
        }


class Trace:
    """Collector for one logical run (one analysis request)."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex
        self.spans: list[Span] = []
        self._token = None

    def __enter__(self) -> Trace:
        self._token = ctx.set_run_id(self.run_id)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            ctx.reset_run_id(self._token)

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs: Any) -> Iterator[Span]:
        span = Span(
            name=name,
            span_id=uuid.uuid4().hex[:16],
            parent_id=ctx.current_span_id(),
            run_id=self.run_id,
            kind=kind,
            attributes=dict(attrs),
        )
        self.spans.append(span)
        token = ctx.set_span_id(span.span_id)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            span.end_ns = time.perf_counter_ns()
            log.warning("span failed", extra={"span": name, "error": span.error})
            raise
        else:
            span.end_ns = time.perf_counter_ns()
        finally:
            ctx.reset_span_id(token)
            if span.end_ns is None:
                span.end_ns = time.perf_counter_ns()

    @property
    def total_ms(self) -> float:
        if not self.spans:
            return 0.0
        return max(s.duration_ms for s in self.spans if s.parent_id is None) if any(
            s.parent_id is None for s in self.spans
        ) else sum(s.duration_ms for s in self.spans)

    def summary(self) -> dict[str, Any]:
        by_name: dict[str, dict[str, Any]] = {}
        for span in self.spans:
            entry = by_name.setdefault(span.name, {"count": 0, "total_ms": 0.0, "errors": 0})
            entry["count"] += 1
            entry["total_ms"] = round(entry["total_ms"] + span.duration_ms, 2)
            entry["errors"] += int(span.status != "ok")
        return {
            "run_id": self.run_id,
            "span_count": len(self.spans),
            "total_ms": round(self.total_ms, 2),
            "by_name": by_name,
        }

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]


class NullTrace(Trace):
    """Trace that records nothing - used on hot paths and in unit tests."""

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs: Any) -> Iterator[Span]:
        yield Span(name=name, span_id="0" * 16, parent_id=None, run_id=self.run_id, kind=kind)
