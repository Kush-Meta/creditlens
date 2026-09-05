"""Logging, tracing and metrics."""
from __future__ import annotations

import json
import logging

import pytest

from creditlens.observability import METRICS, MetricsRegistry, NullTrace, Trace
from creditlens.observability.context import current_run_id
from creditlens.observability.logging_setup import JsonFormatter


class TestTracing:
    def test_spans_nest_and_record_duration(self):
        trace = Trace()
        with trace:
            with trace.span("outer") as outer:
                outer.set(ticker="NVCR")
                with trace.span("inner"):
                    pass
        assert len(trace.spans) == 2
        inner = next(s for s in trace.spans if s.name == "inner")
        outer = next(s for s in trace.spans if s.name == "outer")
        assert inner.parent_id == outer.span_id
        assert outer.attributes["ticker"] == "NVCR"
        assert outer.duration_ms >= 0

    def test_run_id_is_ambient_inside_the_trace(self):
        trace = Trace()
        assert current_run_id() != trace.run_id
        with trace:
            assert current_run_id() == trace.run_id
        assert current_run_id() != trace.run_id

    def test_failed_span_records_the_error_and_reraises(self):
        trace = Trace()
        with pytest.raises(ValueError):
            with trace:
                with trace.span("boom"):
                    raise ValueError("kaboom")
        span = trace.spans[0]
        assert span.status == "error"
        assert "kaboom" in span.error

    def test_summary_groups_by_name(self):
        trace = Trace()
        with trace:
            for _ in range(3):
                with trace.span("tool.compute_ratios"):
                    pass
        summary = trace.summary()
        assert summary["by_name"]["tool.compute_ratios"]["count"] == 3
        assert summary["span_count"] == 3

    def test_otlp_shape(self):
        trace = Trace()
        with trace:
            with trace.span("x", kind="tool"):
                pass
        payload = trace.spans[0].to_otlp_like()
        assert {"traceId", "spanId", "name", "attributes"} <= set(payload)

    def test_null_trace_records_nothing(self):
        trace = NullTrace()
        with trace.span("ignored"):
            pass
        assert trace.spans == []


class TestMetrics:
    def test_counters_and_labels(self):
        registry = MetricsRegistry()
        registry.inc("calls", tool="a")
        registry.inc("calls", tool="a")
        registry.inc("calls", tool="b")
        snapshot = registry.snapshot()["counters"]["calls"]
        assert snapshot["tool=a"] == 2

    def test_histogram_percentiles(self):
        registry = MetricsRegistry()
        for value in range(1, 101):
            registry.observe("latency", float(value))
        summary = registry.snapshot()["histograms"]["latency"]["_"]
        assert summary["count"] == 100
        assert summary["p50"] == pytest.approx(50, abs=2)
        assert summary["max"] == 100

    def test_prometheus_exposition_is_well_formed(self):
        registry = MetricsRegistry()
        registry.describe("calls", "Tool calls")
        registry.inc("calls", tool="a")
        registry.observe("latency", 12.0, tool="a")
        registry.gauge("corpus", 42)
        body = registry.prometheus()
        assert '# HELP calls Tool calls' in body
        assert 'calls{tool="a"} 1.0' in body
        assert 'latency_bucket{tool="a",le="+Inf"} 1' in body
        assert "corpus{} 42" in body or "corpus 42" in body

    def test_reset(self):
        registry = MetricsRegistry()
        registry.inc("x")
        registry.reset()
        assert registry.snapshot()["counters"] == {}

    def test_global_registry_declares_the_core_series(self):
        assert "creditlens_analysis_total" in METRICS._help


class TestLogging:
    def test_json_formatter_emits_parseable_lines_with_context(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1, "hello", None, None
        )
        record.ticker = "NVCR"
        trace = Trace()
        with trace:
            payload = json.loads(formatter.format(record))
        assert payload["msg"] == "hello"
        assert payload["ticker"] == "NVCR"
        assert payload["run_id"] == trace.run_id

    def test_non_serializable_extras_do_not_break_logging(self):
        formatter = JsonFormatter()
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "m", None, None)
        record.weird = object()
        assert json.loads(formatter.format(record))["weird"]


class TestConfig:
    def test_fingerprint_changes_with_answer_affecting_settings(self):
        from creditlens.config import Settings

        base = Settings()
        assert base.fingerprint() == Settings().fingerprint()
        changed = Settings(dense_weight=0.9)
        assert changed.fingerprint() != base.fingerprint()

    def test_fingerprint_ignores_cosmetic_settings(self):
        from creditlens.config import Settings

        assert Settings(log_level="DEBUG").fingerprint() == Settings().fingerprint()
