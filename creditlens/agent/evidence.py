"""The evidence ledger.

Everything the model is allowed to assert has to enter through here. Two
kinds of evidence are tracked:

* **Numeric evidence** - every value produced by the deterministic finance
  layer, with its unit, period, formula and provenance. The verifier later
  extracts numbers from the model's prose and checks each one against this
  ledger; a number that is not in the ledger is, by construction, one the
  model made up.
* **Passage evidence** - retrieved filing text, assigned a stable citation
  label (`C1`, `C2`, ...) that the model is instructed to cite. A citation
  the ledger does not know about is invalid.

This is what makes "evidence-backed" a checkable property rather than a
prompt instruction.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from creditlens.retrieval.hybrid import RetrievedChunk


@dataclass
class NumericEvidence:
    key: str
    label: str
    value: float
    unit: str
    ticker: str | None = None
    period: str | None = None
    formula: str | None = None
    source: str = "computed"
    tool: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "ticker": self.ticker,
            "period": self.period,
            "formula": self.formula,
            "source": self.source,
            "tool": self.tool,
            "is_synthetic": self.is_synthetic,
        }


@dataclass
class PassageEvidence:
    label: str  # C1, C2, ...
    chunk_id: int
    ticker: str
    company: str
    form_type: str
    period: str
    item: str | None
    section: str | None
    accession: str
    filing_date: str
    url: str | None
    text: str
    snippet: str
    score: float
    is_synthetic: bool = False

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "chunk_id": self.chunk_id,
            "ticker": self.ticker,
            "company": self.company,
            "form_type": self.form_type,
            "period": self.period,
            "source": f"{self.form_type} {self.period}".strip(),
            "item": self.item,
            "section": self.section,
            "accession": self.accession,
            "filing_date": self.filing_date,
            "url": self.url,
            "snippet": self.snippet,
            "relevance": round(self.score, 5),
            "is_synthetic": self.is_synthetic,
        }
        if include_text:
            payload["text"] = self.text
        return payload


class EvidenceLedger:
    def __init__(self) -> None:
        self.numbers: dict[str, NumericEvidence] = {}
        self.passages: dict[str, PassageEvidence] = {}
        self._by_chunk: dict[int, str] = {}
        self._counter = 0

    # -- numeric ----------------------------------------------------------
    def add_number(self, evidence: NumericEvidence) -> NumericEvidence:
        self.numbers[evidence.key] = evidence
        return evidence

    def add_numbers(self, items: Iterable[NumericEvidence]) -> None:
        for item in items:
            self.add_number(item)

    def numeric_values(self) -> list[NumericEvidence]:
        return list(self.numbers.values())

    # -- passages ---------------------------------------------------------
    def add_passage(self, hit: RetrievedChunk) -> PassageEvidence:
        existing_label = self._by_chunk.get(hit.record.chunk_id)
        if existing_label:
            return self.passages[existing_label]
        self._counter += 1
        label = f"C{self._counter}"
        record = hit.record
        passage = PassageEvidence(
            label=label,
            chunk_id=record.chunk_id,
            ticker=record.ticker,
            company=record.company_name,
            form_type=record.form_type,
            period=f"{record.fiscal_period or ''}{record.fiscal_year or ''}".strip(),
            item=record.item_code,
            section=record.section_title,
            accession=record.accession,
            filing_date=record.filing_date,
            url=record.url,
            text=record.text,
            snippet=hit.snippet,
            score=hit.score,
            is_synthetic=record.is_synthetic,
        )
        self.passages[label] = passage
        self._by_chunk[record.chunk_id] = label
        return passage

    def known_labels(self) -> set[str]:
        return set(self.passages)

    def citations(self, labels: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(labels) if labels is not None else set(self.passages)
        return [
            self.passages[label].to_dict()
            for label in sorted(wanted, key=_label_sort)
            if label in self.passages
        ]

    def has_synthetic(self) -> bool:
        return any(p.is_synthetic for p in self.passages.values()) or any(
            n.is_synthetic for n in self.numbers.values()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "numeric_facts": len(self.numbers),
            "passages": len(self.passages),
            "tickers": sorted({
                *(n.ticker for n in self.numbers.values() if n.ticker),
                *(p.ticker for p in self.passages.values()),
            }),
            "contains_synthetic": self.has_synthetic(),
        }


def _label_sort(label: str) -> tuple[int, str]:
    digits = "".join(ch for ch in label if ch.isdigit())
    return (int(digits) if digits else 0, label)
