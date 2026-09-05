"""Metric implementations.

Kept separate from the runner so each metric is unit-testable against
hand-built inputs, which matters: a silently wrong eval metric is worse than
no eval at all.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


def recall_at_k(retrieved_ids: Sequence[int], relevant_ids: set[int], k: int | None = None) -> float:
    if not relevant_ids:
        return float("nan")
    top = list(retrieved_ids)[: k or len(retrieved_ids)]
    return len(set(top) & relevant_ids) / len(relevant_ids)


def recall_ceiling_normalized(
    retrieved_ids: Sequence[int], relevant_ids: set[int], k: int
) -> float:
    """Recall@k divided by the best recall@k achievable.

    With 300 relevant passages and k=8, raw recall@k cannot exceed 0.027, so
    comparing raw recall across cases with very different label-set sizes is
    meaningless. Dividing by the ceiling min(1, k/|relevant|) makes the metric
    comparable: 1.0 means every slot that could have held a relevant passage
    did.
    """
    if not relevant_ids:
        return float("nan")
    ceiling = min(1.0, k / len(relevant_ids))
    if ceiling == 0:
        return float("nan")
    return recall_at_k(retrieved_ids, relevant_ids, k) / ceiling


def precision_at_k(retrieved_ids: Sequence[int], relevant_ids: set[int], k: int | None = None) -> float:
    top = list(retrieved_ids)[: k or len(retrieved_ids)]
    if not top:
        return 0.0
    return len(set(top) & relevant_ids) / len(top)


def reciprocal_rank(retrieved_ids: Sequence[int], relevant_ids: set[int]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[int], relevant_ids: set[int], k: int = 10) -> float:
    """Binary-gain nDCG. Ideal DCG assumes all relevant docs rank first."""
    if not relevant_ids:
        return float("nan")
    gains = [1.0 if cid in relevant_ids else 0.0 for cid in list(retrieved_ids)[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_n = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg else float("nan")


def set_f1(predicted: Iterable[str], expected: Iterable[str]) -> dict[str, float]:
    predicted_set, expected_set = set(predicted), set(expected)
    if not expected_set:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    overlap = len(predicted_set & expected_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(expected_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def aggregate(values: Sequence[float]) -> dict[str, float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "n": len(clean),
        "mean": round(sum(clean) / len(clean), 4),
        "p50": round(percentile(clean, 0.50), 4),
        "p95": round(percentile(clean, 0.95), 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
    }


@dataclass
class MetricAccumulator:
    """Collects per-case values and renders the summary table."""

    values: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, value: float | None) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        self.values.setdefault(name, []).append(float(value))

    def bump(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            name: aggregate(series) for name, series in sorted(self.values.items())
        }
        out["counts"] = dict(sorted(self.counts.items()))
        return out


def compare_metric_values(
    reported: str | float | None, truth: float | None, tolerance_pct: float = 1.0
) -> tuple[bool, float | None]:
    """Compare a reported metric string ("3.42x", "18.6%") with ground truth."""
    if truth is None or reported is None:
        return False, None
    if isinstance(reported, str):
        cleaned = (
            reported.replace(",", "").replace("$", "").replace("x", "")
            .replace("%", "").replace("days", "").strip()
        )
        try:
            value = float(cleaned)
        except ValueError:
            return False, None
    else:
        value = float(reported)
    # A value written as "0.21x" asserts 0.21 +/- 0.005. Judging it against a
    # 1% relative tolerance would fail a correctly-rounded number, so the
    # written precision sets a floor on what counts as agreement.
    precision_tolerance = 0.0
    if isinstance(reported, str) and "." in cleaned:
        precision_tolerance = 0.5 * (10 ** -len(cleaned.split(".", 1)[1]))
    absolute = abs(value - truth)
    if absolute <= precision_tolerance:
        return True, 0.0
    if truth == 0:
        return abs(value) < 1e-9, abs(value)
    deviation = absolute / abs(truth) * 100.0
    return deviation <= tolerance_pct, deviation
