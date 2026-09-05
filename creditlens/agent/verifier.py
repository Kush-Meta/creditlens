"""Post-hoc verification of the model's answer.

Prompting a model to "only use numbers from tools" is a request, not a
guarantee. This module turns it into a measurement:

* **Numeric grounding.** Every number in the narrative is extracted with its
  unit and matched against the evidence ledger within tolerance. Numbers that
  match are `verified`; numbers that match nothing are `unsupported`; numbers
  that are close to a ledger value for the same metric but outside tolerance
  are `contradicted` - the most serious class, because they look plausible.
* **Citation validity.** Every `[C#]` label must exist in the ledger.
* **Confidence.** Derived from those two rates plus evidence breadth, and
  capped by the model's own stated confidence.

The output feeds both the API response and the eval harness, so the same
definition of "unsupported claim" is used in production and in measurement.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from creditlens.agent.evidence import EvidenceLedger, NumericEvidence
from creditlens.config import get_settings
from creditlens.observability import get_logger

log = get_logger(__name__)

#: Numbers with a unit we can interpret. Ordered: most specific pattern first.
_NUMBER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("usd_scaled", re.compile(
        r"\$\s?(-?\d[\d,]*\.?\d*)\s*(billion|bn|million|mm|m|thousand|k|trillion)\b", re.I)),
    ("usd", re.compile(r"\$\s?(-?\d[\d,]*\.?\d*)")),
    ("pp", re.compile(r"(-?\d[\d,]*\.?\d*)\s*(?:percentage points?|pp)\b", re.I)),
    ("bps", re.compile(r"(-?\d[\d,]*\.?\d*)\s*(?:bps|basis points)\b", re.I)),
    ("percent", re.compile(r"(-?\d[\d,]*\.?\d*)\s?%")),
    ("multiple", re.compile(r"(-?\d[\d,]*\.?\d*)\s?x\b", re.I)),
    ("days", re.compile(r"(-?\d[\d,]*\.?\d*)\s*days\b", re.I)),
)

_SCALES = {
    "trillion": 1e12, "billion": 1e9, "bn": 1e9,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

_CITATION_RE = re.compile(r"\[(C\d+)\]")

#: Numbers that are years, counts or ordinals, not financial claims.
_IGNORE_CONTEXT = re.compile(
    r"(19|20)\d{2}|\bQ[1-4]\b|\bFY\b|\bitem\b|\bpage\b", re.I
)


@dataclass
class NumericClaim:
    text: str
    value: float
    unit: str
    context: str
    #: the single sentence the number appears in, used to decide *which*
    #: metric the claim is about; the wider `context` is kept for display
    #: and for matching quotations against retrieved passages
    sentence: str = ""
    #: the comma-delimited clause around the number. A sentence that lists
    #: several metrics ("Current ratio 1.09x, Cash / Debt 42.3%, FCF margin
    #: 22.7%") names them all, so sentence-level scope attributes each figure
    #: to whichever metric happens to score highest. The clause disambiguates.
    clause: str = ""
    #: absolute half-ulp implied by how the number was written, e.g. "2.7%"
    #: was printed to one decimal and therefore means 2.7 +/- 0.05
    precision_tolerance: float = 0.0
    #: verified   - matches a computed value from the evidence ledger
    #: cited       - not computed, but quoted from a retrieved filing passage
    #: contradicted- near a computed value but outside tolerance
    #: unsupported - traceable to nothing
    status: str = "unsupported"
    matched_key: str | None = None
    matched_label: str | None = None
    matched_value: float | None = None
    deviation_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.text,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "matched_evidence": self.matched_label,
            "matched_key": self.matched_key,
            "matched_value": self.matched_value,
            "deviation_pct": None if self.deviation_pct is None else round(self.deviation_pct, 3),
            "precision_tolerance": self.precision_tolerance,
            "context": self.context,
        }


@dataclass
class VerificationReport:
    numeric_claims: list[NumericClaim] = field(default_factory=list)
    valid_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    uncited_passages: list[str] = field(default_factory=list)
    confidence: str = "low"
    stated_confidence: str | None = None
    notes: list[str] = field(default_factory=list)
    contains_synthetic: bool = False

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.numeric_claims if c.status == "verified")

    @property
    def cited_count(self) -> int:
        return sum(1 for c in self.numeric_claims if c.status == "cited")

    @property
    def unsupported_count(self) -> int:
        return sum(1 for c in self.numeric_claims if c.status == "unsupported")

    @property
    def contradicted_count(self) -> int:
        return sum(1 for c in self.numeric_claims if c.status == "contradicted")

    @property
    def numeric_accuracy(self) -> float:
        """Share of figures traceable to computation or to quoted filing text.

        Figures quoted from a cited passage ("$2.1 billion of senior notes
        mature within 24 months [C3]") are grounded by the citation, not by the
        calculation engine, and counting them as errors would penalise exactly
        the behaviour we want.
        """
        total = len(self.numeric_claims)
        if total == 0:
            return 1.0
        return (self.verified_count + self.cited_count) / total

    @property
    def unsupported_rate(self) -> float:
        total = len(self.numeric_claims)
        if total == 0:
            return 0.0
        return (self.unsupported_count + self.contradicted_count) / total

    @property
    def citation_validity(self) -> float:
        total = len(self.valid_citations) + len(self.invalid_citations)
        return 1.0 if total == 0 else len(self.valid_citations) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "model_stated_confidence": self.stated_confidence,
            "numeric_claims_checked": len(self.numeric_claims),
            "numeric_claims_verified": self.verified_count,
            "numeric_claims_quoted_from_filings": self.cited_count,
            "numeric_claims_unsupported": self.unsupported_count,
            "numeric_claims_contradicted": self.contradicted_count,
            "numeric_accuracy": round(self.numeric_accuracy, 4),
            "unsupported_claim_rate": round(self.unsupported_rate, 4),
            "citation_validity": round(self.citation_validity, 4),
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "unused_retrieved_passages": self.uncited_passages,
            "claims": [c.to_dict() for c in self.numeric_claims],
            "notes": self.notes,
            "contains_synthetic_data": self.contains_synthetic,
        }


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    claims: list[NumericClaim] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in consumed)

    for unit, pattern in _NUMBER_PATTERNS:
        for match in pattern.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            raw = match.group(1).replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue
            context = text[max(0, match.start() - 90): match.end() + 60].replace("\n", " ")
            sentence = _sentence_around(text, match.start(), match.end())
            clause = _sentence_around(text, match.start(), match.end(), _CLAUSE_BREAK)
            if unit == "usd" and _IGNORE_CONTEXT.search(match.group(0)):
                continue
            if unit == "usd_scaled":
                scale = _SCALES.get(match.group(2).lower(), 1.0)
                value *= scale
            consumed.append((match.start(), match.end()))
            claims.append(NumericClaim(
                text=match.group(0).strip(), value=value,
                unit="USD" if unit.startswith("usd") else unit, context=context.strip(),
                sentence=sentence, clause=clause,
                precision_tolerance=_half_ulp(raw) * (
                    _SCALES.get(match.group(2).lower(), 1.0) if unit == "usd_scaled" else 1.0
                ),
            ))
    return claims


_SENTENCE_BREAK = re.compile(r"[.!?;\n]")
_CLAUSE_BREAK = re.compile(r"[.!?;,\n]")


def _sentence_around(
    text: str, start: int, end: int, breaker: re.Pattern[str] = _SENTENCE_BREAK
) -> str:
    left = 0
    for match in breaker.finditer(text[:start]):
        left = match.end()
    right_match = breaker.search(text, end)
    right = right_match.start() if right_match else len(text)
    return " ".join(text[left:right].split())


def _half_ulp(raw: str) -> float:
    """Absolute uncertainty implied by the written precision of a number.

    "18.6%" asserts 18.6 +/- 0.05, not 18.600000. Without this, a value the
    calculation engine produced as 18.63 and the narrative faithfully rounded
    to 18.6 would be scored as a contradiction - the single largest source of
    false positives in numeric verification.
    """
    if "." in raw:
        decimals = len(raw.split(".", 1)[1])
        return 0.5 * (10 ** -decimals)
    return 0.5


def _candidate_values(evidence: NumericEvidence) -> list[tuple[float, str]]:
    """Values a claim may legitimately match, accounting for reporting scale.

    XBRL reports absolute dollars; analysts write "$4.2 billion". Both forms
    are accepted, as is the millions convention used by many filers.
    """
    value = evidence.value
    out = [(value, "as-reported")]
    if evidence.unit == "USD":
        out += [(value * 1e6, "reported-in-millions"), (value * 1e9, "reported-in-billions")]
    if evidence.unit in {"%", "pp"}:
        out.append((value * 100, "fraction-to-bps"))
    return out


#: Analyst vocabulary that never appears in a ratio's label but unambiguously
#: names it. A phrase hit is worth two points - enough on its own to pin a
#: claim to one metric - because these phrases are far more specific than any
#: single label word.
_METRIC_PHRASES: dict[str, tuple[str, ...]] = {
    "debt_to_ebitda": ("gross leverage", "leverage ratio", "debt to ebitda",
                        "debt/ebitda", "leverage stands", "leverage of",
                        "leverage was", "leverage is", "leverage rose",
                        "leverage fell"),
    "net_debt_to_ebitda": ("net leverage", "net debt to ebitda", "net debt/ebitda"),
    "ebitda_interest_coverage": ("interest coverage", "coverage ratio",
                                  "covered interest", "ebitda coverage"),
    "ebit_interest_coverage": ("ebit coverage", "ebit interest"),
    "fcf_interest_coverage": ("fcf coverage", "free cash flow coverage"),
    "current_ratio": ("current ratio",),
    "quick_ratio": ("quick ratio", "acid test"),
    "cash_ratio": ("cash ratio",),
    "cash_to_debt": ("cash to debt", "cash/debt"),
    "operating_margin": ("operating margin",),
    "gross_margin": ("gross margin",),
    "ebitda_margin": ("ebitda margin",),
    "net_margin": ("net margin", "profit margin"),
    "fcf_margin": ("free cash flow margin", "fcf margin"),
    "cfo_to_debt": ("cash flow to debt", "cfo to debt", "cfo/debt"),
    "return_on_assets": ("return on assets", "roa"),
    "return_on_equity": ("return on equity", "roe"),
    "debt_to_equity": ("debt to equity", "debt/equity", "gearing"),
    "debt_to_capital": ("debt to capital", "debt/capital"),
}

_PHRASE_INDEX: dict[str, tuple[str, ...]] = _METRIC_PHRASES

_LABEL_STOP = {
    "the", "of", "and", "to", "a", "in", "per", "over", "from", "change",
    "total", "ttm", "fy", "q1", "q2", "q3", "q4", "window", "score",
}


def _label_tokens(label: str) -> set[str]:
    raw = re.findall(r"[a-z]{3,}", label.lower())
    return {t for t in raw if t not in _LABEL_STOP}


def _affinity(item: NumericEvidence, context: str) -> int:
    """How strongly the claim's wording points at this specific metric.

    Without this, "the current ratio was 1.29x" happily matches a debt/equity
    of 1.29 and is scored as verified - a false pass that would make the whole
    verification layer meaningless. With it, a claim is matched only against
    the metric its own sentence names.
    """
    lowered = context.lower()
    score = 0
    metric_key = item.key.rsplit(":", 1)[-1]
    for phrase in _PHRASE_INDEX.get(metric_key, ()):
        if phrase in lowered:
            score += 2
            break
    for token in _label_tokens(item.label):
        if token in lowered:
            score += 1
    if item.ticker and item.ticker.lower() in lowered:
        score += 1
    return score


#: Below this, the sentence has not named a metric specifically enough to
#: exclude anything - a lone generic word like "gross" or "revenue" must not
#: narrow the candidate pool.
_AFFINITY_RESTRICT_THRESHOLD = 2


def verify_numbers(
    claims: Iterable[NumericClaim], ledger: EvidenceLedger, tolerance_pct: float
) -> list[NumericClaim]:
    evidence = ledger.numeric_values()
    checked: list[NumericClaim] = []
    for claim in claims:
        # When the surrounding prose names a metric, only that metric's
        # evidence is eligible; otherwise every unit-compatible value is.
        # Prefer the tightest scope that actually names a metric: the clause
        # first, widening to the sentence only if the clause names nothing.
        pool = evidence
        for scope in (claim.clause, claim.sentence or claim.context):
            if not scope:
                continue
            affinities = {id(item): _affinity(item, scope) for item in evidence}
            peak = max(affinities.values(), default=0)
            if peak >= _AFFINITY_RESTRICT_THRESHOLD:
                pool = [i for i in evidence if affinities[id(i)] == peak]
                break

        best: tuple[float, NumericEvidence, float] | None = None
        for item in pool:
            if not _unit_compatible(claim.unit, item.unit):
                continue
            for candidate, _mode in _candidate_values(item):
                absolute = abs(claim.value - candidate)
                # A hair of slack: |1.1 - 1.05| evaluates to 0.05000000000000004
                # in binary floating point, which would fail an exact <= against
                # a 0.05 tolerance and report a correctly-rounded figure as a
                # contradiction.
                if absolute <= claim.precision_tolerance * (1 + 1e-9) + 1e-12:
                    deviation = 0.0
                elif candidate == 0:
                    continue
                else:
                    deviation = absolute / abs(candidate) * 100.0
                if best is None or deviation < best[0]:
                    best = (deviation, item, candidate)
        if best is None:
            claim.status = "unsupported"
        else:
            deviation, item, candidate = best
            claim.matched_key = item.key
            claim.matched_label = item.label
            claim.matched_value = candidate
            claim.deviation_pct = deviation
            if deviation <= tolerance_pct:
                claim.status = "verified"
            elif deviation <= 25.0:
                # Close to a real value but outside tolerance: a transcription or
                # rounding error, which is worse than an unmatched number.
                claim.status = "contradicted"
            else:
                claim.status = "unsupported"
                claim.matched_key = None
                claim.matched_label = None
                claim.matched_value = None
                claim.deviation_pct = None
        checked.append(claim)
    return checked


def ground_in_passages(
    claims: Iterable[NumericClaim], ledger: EvidenceLedger
) -> None:
    """Re-classify unmatched figures that are quoted from a retrieved passage.

    Matching requires the numeric token AND a slice of its surrounding context
    to appear in the same passage, so an incidental digit collision does not
    launder an invented figure.
    """
    corpus = {
        label: " ".join(passage.text.split()).lower()
        for label, passage in ledger.passages.items()
    }
    if not corpus:
        return
    for claim in claims:
        # Both unsupported and contradicted claims are re-checked: a figure
        # quoted verbatim from a cited filing is a quotation, even when it
        # happens to sit within 25% of some computed value.
        if claim.status not in {"unsupported", "contradicted"}:
            continue
        token = claim.text.strip().lower()
        context = " ".join(claim.context.split()).lower()
        for label, text in corpus.items():
            if token not in text:
                continue
            if _context_overlaps(context, text, token):
                claim.status = "cited"
                claim.matched_label = f"quoted from passage {label}"
                claim.matched_key = label
                claim.matched_value = None
                claim.deviation_pct = None
                break


def _context_overlaps(context: str, passage: str, token: str, min_words: int = 4) -> bool:
    """True when the number appears in the passage with the same wording around it.

    Checked on both sides independently: an analyst quoting a filing usually
    keeps the words that follow the figure ("... of senior notes mature
    within") while replacing the words before it ("Management notes ..."), so
    requiring a prefix match would reject genuine quotations.
    """
    position = context.find(token)
    if position < 0:
        return False
    before = context[:position].split()
    after = context[position + len(token):].split()

    for words in (after, list(reversed(before))):
        for length in range(min(6, len(words)), min_words - 1, -1):
            window = words[:length] if words is after else list(reversed(words[:length]))
            phrase = " ".join(window).strip(" .,;:")
            if len(phrase) < 12:
                continue
            probe = f"{token} {phrase}" if words is after else f"{phrase} {token}"
            if probe in passage:
                return True
            if phrase in passage:
                return True
    return False


def _unit_compatible(claim_unit: str, evidence_unit: str) -> bool:
    if claim_unit == evidence_unit:
        return True
    numeric_like = {"multiple", "x", "score", ""}
    if claim_unit == "multiple" and evidence_unit in {"x", "score"}:
        return True
    if claim_unit in {"pp", "bps"} and evidence_unit in {"%", "pp"}:
        return True
    if claim_unit == "percent" and evidence_unit == "%":
        return True
    if claim_unit == "days" and evidence_unit == "days":
        return True
    if claim_unit == "USD" and evidence_unit == "USD":
        return True
    return claim_unit in numeric_like and evidence_unit in numeric_like


def verify(
    *,
    narrative: str,
    ledger: EvidenceLedger,
    stated_confidence: str | None = None,
    declared_citations: Iterable[str] | None = None,
) -> VerificationReport:
    settings = get_settings()
    report = VerificationReport(stated_confidence=stated_confidence)
    report.contains_synthetic = ledger.has_synthetic()

    report.numeric_claims = verify_numbers(
        extract_numeric_claims(narrative), ledger, settings.numeric_tolerance_pct
    )
    ground_in_passages(report.numeric_claims, ledger)

    known = ledger.known_labels()
    used = set(_CITATION_RE.findall(narrative)) | {
        c.strip() for c in (declared_citations or []) if c and c.strip()
    }
    report.valid_citations = sorted(used & known, key=_label_num)
    report.invalid_citations = sorted(used - known, key=_label_num)
    report.uncited_passages = sorted(known - used, key=_label_num)

    report.confidence = _confidence(report, ledger)
    if report.invalid_citations:
        report.notes.append(
            f"answer cited {len(report.invalid_citations)} label(s) that were never "
            f"retrieved: {', '.join(report.invalid_citations)}"
        )
    if report.contradicted_count:
        report.notes.append(
            f"{report.contradicted_count} figure(s) are close to a computed value but "
            "outside tolerance - likely transcription or rounding errors"
        )
    if report.unsupported_count:
        report.notes.append(
            f"{report.unsupported_count} figure(s) could not be traced to any tool "
            "output or retrieved passage"
        )
    if report.cited_count:
        report.notes.append(
            f"{report.cited_count} figure(s) are quoted from cited filing text rather "
            "than computed"
        )
    if report.contains_synthetic:
        report.notes.append(
            "answer draws on synthetic demo data; figures are not filed results"
        )
    return report


def _label_num(label: str) -> tuple[int, str]:
    digits = "".join(ch for ch in label if ch.isdigit())
    return (int(digits) if digits else 0, label)


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_CONFIDENCE = {0: "low", 1: "medium", 2: "high"}


def _confidence(report: VerificationReport, ledger: EvidenceLedger) -> str:
    settings = get_settings()
    if report.contradicted_count or report.invalid_citations:
        return "low"
    if report.unsupported_rate > 0.25:
        return "low"

    score = 2
    if report.unsupported_rate > 0.05:
        score -= 1
    if len(ledger.numbers) < 4:
        score -= 1
    if len(ledger.passages) < settings.min_citations_for_high_confidence:
        score -= 1
    score = max(0, min(2, score))

    if report.stated_confidence in _CONFIDENCE_RANK:
        # The model may lower confidence but never raise it above what the
        # evidence supports.
        score = min(score, _CONFIDENCE_RANK[report.stated_confidence])
    return _RANK_CONFIDENCE[score]
