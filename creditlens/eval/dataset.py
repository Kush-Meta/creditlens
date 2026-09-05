"""Eval case definitions.

Relevance is labelled *programmatically* rather than by hand: a case declares
the issuer, the filing items and the terms a genuinely useful passage must
contain, and the harness resolves that predicate against the whole corpus to
build the ground-truth relevant set. That makes recall well-defined (we know
the denominator), keeps labels consistent as the corpus grows, and costs
nothing to re-derive after a re-ingest.

The trade-off is honest and worth stating: programmatic labels measure
*retrievability of the passages we specified*, not human-judged relevance.
They are excellent for regression-testing a retrieval change and weaker as an
absolute quality score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass
class RelevanceSpec:
    """Predicate defining which corpus chunks count as relevant for a case."""

    tickers: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    form_types: list[str] = field(default_factory=list)
    #: a chunk must contain at least one term from each group (AND of ORs)
    any_of: list[list[str]] = field(default_factory=list)

    def matches(self, record: Any) -> bool:
        if self.tickers and record.ticker.upper() not in {t.upper() for t in self.tickers}:
            return False
        if self.items and (record.item_code or "").upper() not in {i.upper() for i in self.items}:
            return False
        if self.form_types and record.form_type.upper() not in {
            f.upper() for f in self.form_types
        }:
            return False
        text = record.text.lower()
        return all(any(term.lower() in text for term in group) for group in self.any_of)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tickers": self.tickers, "items": self.items,
            "form_types": self.form_types, "any_of": self.any_of,
        }


@dataclass
class EvalCase:
    id: str
    question: str
    category: str
    tickers: list[str] = field(default_factory=list)
    #: tools a competent run should use; scored as set F1, not exact match
    expected_tools: list[str] = field(default_factory=list)
    #: (ticker, period, ratio) triples whose values are recomputed
    #: independently and compared against what the answer reported
    expected_metrics: list[dict[str, str]] = field(default_factory=list)
    expected_direction: str | None = None
    relevance: RelevanceSpec | None = None
    retrieval_query: str | None = None
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    requires_synthetic: bool = False
    #: slice dimensions, so results can be read per sector and per credit profile
    sector: str | None = None
    profile: str | None = None
    generated: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalCase:
        relevance = payload.get("relevance")
        return cls(
            id=payload["id"],
            question=payload["question"],
            category=payload.get("category", "general"),
            tickers=payload.get("tickers", []),
            expected_tools=payload.get("expected_tools", []),
            expected_metrics=payload.get("expected_metrics", []),
            expected_direction=payload.get("expected_direction"),
            relevance=RelevanceSpec(**relevance) if relevance else None,
            retrieval_query=payload.get("retrieval_query"),
            must_mention=payload.get("must_mention", []),
            must_not_mention=payload.get("must_not_mention", []),
            requires_synthetic=payload.get("requires_synthetic", False),
            sector=payload.get("sector"),
            profile=payload.get("profile"),
            generated=payload.get("generated", False),
            notes=payload.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "question": self.question, "category": self.category,
            "tickers": self.tickers, "expected_tools": self.expected_tools,
            "expected_metrics": self.expected_metrics,
            "expected_direction": self.expected_direction,
            "relevance": self.relevance.to_dict() if self.relevance else None,
            "retrieval_query": self.retrieval_query,
            "must_mention": self.must_mention,
            "must_not_mention": self.must_not_mention,
            "requires_synthetic": self.requires_synthetic,
            "sector": self.sector,
            "profile": self.profile,
            "generated": self.generated,
            "notes": self.notes,
        }


def load_suite(name: str = "golden", *, directory: Path | None = None) -> list[EvalCase]:
    """Load one suite, or several comma-separated ones ("golden,universe").

    Case ids must be unique across combined suites: a duplicate would be scored
    twice and silently skew the aggregate.
    """
    base = directory or DATASET_DIR
    cases: list[EvalCase] = []
    seen: dict[str, str] = {}

    for suite_name in [n.strip() for n in name.split(",") if n.strip()]:
        path = base / f"{suite_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"eval suite not found: {path}")
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                case = EvalCase.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if case.id in seen:
                raise ValueError(
                    f"{path}:{line_no}: duplicate case id {case.id!r} "
                    f"(already defined in {seen[case.id]})"
                )
            seen[case.id] = suite_name
            cases.append(case)
    return cases


def available_suites(directory: Path | None = None) -> list[str]:
    base = directory or DATASET_DIR
    return sorted(p.stem for p in base.glob("*.jsonl"))
