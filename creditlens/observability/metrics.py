"""Process-local metrics registry with a Prometheus text exposition.

Deliberately dependency-free. Counters/histograms are the two shapes that
matter here: request and tool call counts, and latency/cost distributions.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

_DEFAULT_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((labels or {}).items()))


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple, float]] = defaultdict(dict)
        self._gauges: dict[str, dict[tuple, float]] = defaultdict(dict)
        self._hist: dict[str, dict[tuple, list[float]]] = defaultdict(dict)
        self._help: dict[str, str] = {}
        self.started_at = time.time()

    def describe(self, name: str, help_text: str) -> None:
        self._help[name] = help_text

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = _labels_key(labels)
        with self._lock:
            self._counters[name][key] = self._counters[name].get(key, 0.0) + value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[name][_labels_key(labels)] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = _labels_key(labels)
        with self._lock:
            self._hist[name].setdefault(key, []).append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "counters": {
                    name: {_fmt(k): v for k, v in series.items()}
                    for name, series in self._counters.items()
                },
                "gauges": {
                    name: {_fmt(k): v for k, v in series.items()}
                    for name, series in self._gauges.items()
                },
                "histograms": {
                    name: {_fmt(k): _summarize(v) for k, v in series.items()}
                    for name, series in self._hist.items()
                },
            }

    def prometheus(self, buckets: Iterable[float] = _DEFAULT_BUCKETS) -> str:
        lines: list[str] = []
        with self._lock:
            for name, series in self._counters.items():
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} counter")
                for key, value in series.items():
                    lines.append(f"{name}{_prom_labels(key)} {value}")
            for name, series in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                for key, value in series.items():
                    lines.append(f"{name}{_prom_labels(key)} {value}")
            for name, series in self._hist.items():
                lines.append(f"# TYPE {name} histogram")
                for key, values in series.items():
                    cumulative = 0
                    ordered = sorted(values)
                    for bound in buckets:
                        cumulative = sum(1 for v in ordered if v <= bound)
                        lines.append(
                            f"{name}_bucket{_prom_labels(key, le=str(bound))} {cumulative}"
                        )
                    lines.append(f"{name}_bucket{_prom_labels(key, le='+Inf')} {len(ordered)}")
                    lines.append(f"{name}_sum{_prom_labels(key)} {sum(ordered)}")
                    lines.append(f"{name}_count{_prom_labels(key)} {len(ordered)}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._hist.clear()


def _fmt(key: tuple) -> str:
    return ",".join(f"{k}={v}" for k, v in key) or "_"


def _prom_labels(key: tuple, **extra: str) -> str:
    items = list(key) + list(extra.items())
    if not items:
        return ""
    inner = ",".join(f'{k}="{v!s}"' for k, v in items)
    return "{" + inner + "}"


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, round((len(ordered) - 1) * p))
        return round(ordered[idx], 3)
    return {
        "count": len(ordered),
        "sum": round(sum(ordered), 3),
        "min": round(ordered[0], 3),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": round(ordered[-1], 3),
    }


METRICS = MetricsRegistry()
METRICS.describe("creditlens_analysis_total", "Analyses executed, by outcome")
METRICS.describe("creditlens_tool_calls_total", "Agent tool invocations, by tool and outcome")
METRICS.describe("creditlens_llm_tokens_total", "LLM tokens consumed, by direction")
METRICS.describe("creditlens_llm_cost_usd_total", "Estimated LLM spend in USD")
