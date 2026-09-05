"""Generate the universe evaluation suite.

A hand-written suite does not scale past a handful of issuers, and a corpus of
forty-two issuers whose behaviour is measured on four of them is not really
measured at all. This generator produces a case per (issuer, question family)
assignment so coverage tracks the corpus, while the hand-written `golden` suite
keeps the cases that need judgement - causal attribution, data-gap honesty,
issuers that are deliberately absent.

Three rules keep generated cases from becoming filler:

1. **Every case asserts something checkable.** A ratio whose ground truth is
   recomputed independently, a relevance predicate resolved against the corpus,
   or a required tool - never just "did it answer".
2. **Assignment is deterministic**, seeded on the ticker, so regenerating the
   suite produces byte-identical output and results stay comparable.
3. **Cases that cannot be scored are dropped at build time**, not silently
   failed at run time: a relevance predicate that matches nothing measures
   nothing, and is filtered out by `validate_suite`.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from creditlens.eval.dataset import DATASET_DIR, EvalCase, RelevanceSpec
from creditlens.ingest.universe import UNIVERSE, Issuer
from creditlens.observability import get_logger

log = get_logger(__name__)

#: Vocabulary that genuinely appears in the relevant section of essentially any
#: 10-K. Deliberately generic: a narrow sector phrase that a filer happens not
#: to use produces an empty label set, which measures nothing.
DEBT_TERMS = ["debt", "indebtedness", "notes", "borrowings", "credit facility"]
LIQUIDITY_TERMS = ["liquidity", "cash", "credit facility", "capital resources"]
RISK_TERMS = ["adversely", "risk", "could", "uncertain"]
MARGIN_TERMS = ["margin", "cost of", "operating expenses", "gross"]
CASHFLOW_TERMS = ["cash flow", "operating activities", "capital expenditures"]


@dataclass(frozen=True)
class Family:
    """One question family, instantiated per issuer."""

    key: str
    category: str
    question: Callable[[Issuer], str]
    retrieval_query: Callable[[Issuer], str]
    items: list[str]
    any_of: list[list[str]]
    expected_tools: list[str]
    ratio: str | None = None


FAMILIES: tuple[Family, ...] = (
    Family(
        "leverage", "calculation",
        lambda i: f"What is {i.name}'s debt-to-EBITDA ratio, and what does it imply for credit quality?",
        lambda i: f"{i.name} total debt senior notes borrowings outstanding",
        ["7", "1A", "8"], [DEBT_TERMS],
        ["compute_ratios"], ratio="debt_to_ebitda",
    ),
    Family(
        "coverage", "calculation",
        lambda i: f"How much interest coverage does {i.name} have?",
        lambda i: f"{i.name} interest expense debt service",
        ["7", "8"], [["interest"]],
        ["compute_ratios"], ratio="ebitda_interest_coverage",
    ),
    Family(
        "liquidity", "liquidity",
        lambda i: f"How much liquidity does {i.name} have and what backs it?",
        lambda i: f"{i.name} cash and cash equivalents revolving credit facility liquidity",
        ["7"], [LIQUIDITY_TERMS],
        ["compute_ratios", "search_filings"], ratio="current_ratio",
    ),
    Family(
        "risk", "risk",
        lambda i: f"What are the most significant risks {i.name} discloses in its latest annual report?",
        lambda i: f"{i.name} risk factors competition regulation operations",
        ["1A"], [RISK_TERMS],
        ["search_filings"],
    ),
    Family(
        "trend", "trend",
        lambda i: f"How has {i.name}'s leverage evolved over recent periods?",
        lambda i: f"{i.name} indebtedness leverage debt levels capital structure",
        ["7", "1A"], [DEBT_TERMS],
        ["metric_trend"], ratio="net_debt_to_ebitda",
    ),
    Family(
        "margin", "causal",
        lambda i: f"What has driven the change in {i.name}'s operating margin?",
        lambda i: f"{i.name} operating income gross margin cost of revenue drivers",
        ["7"], [MARGIN_TERMS],
        ["compare_periods"], ratio="operating_margin",
    ),
    Family(
        "cashflow", "cash_flow",
        lambda i: f"Is {i.name} generating enough cash to service its obligations?",
        lambda i: f"{i.name} cash provided by operating activities capital expenditures free cash flow",
        ["7"], [CASHFLOW_TERMS],
        ["compute_ratios"], ratio="cfo_to_debt",
    ),
    Family(
        "profile", "profile",
        lambda i: f"Summarize {i.name}'s credit profile.",
        lambda i: f"{i.name} liquidity debt cash flow capital resources",
        ["7", "1A"], [LIQUIDITY_TERMS],
        ["credit_scorecard", "compute_ratios"],
    ),
)

FAMILY_BY_KEY = {family.key: family for family in FAMILIES}

#: Families every issuer gets, because they exercise the metrics that define a
#: credit view. The rest are assigned round-robin so the suite stays balanced
#: without ballooning to |issuers| x |families|.
CORE_FAMILIES = ("leverage", "risk")
ROTATING_FAMILIES = ("liquidity", "trend", "margin", "cashflow", "profile", "coverage")


def _rotation_index(ticker: str, salt: str = "") -> int:
    digest = hashlib.blake2b(f"{ticker}{salt}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def build_issuer_cases(issuer: Issuer, *, rotating_per_issuer: int = 2) -> list[EvalCase]:
    keys = list(CORE_FAMILIES)
    start = _rotation_index(issuer.ticker) % len(ROTATING_FAMILIES)
    for offset in range(rotating_per_issuer):
        keys.append(ROTATING_FAMILIES[(start + offset) % len(ROTATING_FAMILIES)])

    cases: list[EvalCase] = []
    for key in dict.fromkeys(keys):
        family = FAMILY_BY_KEY[key]
        cases.append(EvalCase(
            id=f"u-{issuer.ticker.lower()}-{key}",
            question=family.question(issuer),
            category=family.category,
            tickers=[issuer.ticker],
            expected_tools=list(family.expected_tools),
            expected_metrics=(
                [{"ticker": issuer.ticker, "period": "TTM", "ratio": family.ratio}]
                if family.ratio else []
            ),
            relevance=RelevanceSpec(
                tickers=[issuer.ticker], items=list(family.items),
                any_of=[list(group) for group in family.any_of],
            ),
            retrieval_query=family.retrieval_query(issuer),
            sector=issuer.sector,
            profile=issuer.profile,
            generated=True,
        ))
    return cases


def build_comparison_cases(issuers: Sequence[Issuer]) -> list[EvalCase]:
    """One within-sector comparison per sector, pairing dissimilar profiles.

    Comparison is where a credit tool earns its keep, and where retrieval most
    often fails by returning evidence for only one side. Pairing issuers with
    *different* sampling profiles makes the comparison discriminative rather
    than a match between two lookalikes.
    """
    by_sector: dict[str, list[Issuer]] = {}
    for issuer in issuers:
        by_sector.setdefault(issuer.sector, []).append(issuer)

    cases: list[EvalCase] = []
    for sector, members in sorted(by_sector.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda i: i.ticker)
        pair = _most_dissimilar_pair(ordered)
        if pair is None:
            continue
        left, right = pair
        cases.append(EvalCase(
            id=f"u-cmp-{left.ticker.lower()}-{right.ticker.lower()}",
            question=(
                f"Compare the leverage and liquidity of {left.name} and {right.name}. "
                "Which is the stronger credit?"
            ),
            category="comparison",
            tickers=[left.ticker, right.ticker],
            expected_tools=["compare_companies"],
            expected_metrics=[
                {"ticker": left.ticker, "period": "TTM", "ratio": "debt_to_ebitda"},
                {"ticker": right.ticker, "period": "TTM", "ratio": "debt_to_ebitda"},
            ],
            relevance=RelevanceSpec(
                tickers=[left.ticker, right.ticker], items=["7"],
                any_of=[LIQUIDITY_TERMS],
            ),
            retrieval_query=(
                f"{left.name} {right.name} liquidity debt capital resources"
            ),
            sector=sector,
            profile="mixed",
            generated=True,
            notes="within-sector comparison across differing sampling profiles",
        ))
    return cases


def _most_dissimilar_pair(members: Sequence[Issuer]) -> tuple[Issuer, Issuer] | None:
    """Prefer a pair whose sampling profiles differ; fall back to the first two."""
    for i, left in enumerate(members):
        for right in members[i + 1:]:
            if left.profile != right.profile:
                return left, right
    return (members[0], members[1]) if len(members) >= 2 else None


#: Sampling profiles for the fictional demo issuers, so the same generator can
#: build a network-free suite. CI gates on this one: it needs no EDGAR ingest,
#: yet still exercises every question family end to end.
_FIXTURE_PROFILES: dict[str, tuple[str, str]] = {
    "NVCR": ("Industrials", "leveraged"),
    "ARMT": ("Technology", "net_cash"),
    "KSTR": ("Technology", "leveraged"),
    "HRBG": ("Retail", "cyclical_stressed"),
}


def fixture_issuers() -> list[Issuer]:
    """The synthetic demo issuers, adapted to the generator's issuer shape."""
    from creditlens.ingest.fixture_data import ISSUERS

    out: list[Issuer] = []
    for spec in ISSUERS:
        sector, profile = _FIXTURE_PROFILES.get(spec.ticker, (spec.industry, "leveraged"))
        out.append(Issuer(
            ticker=spec.ticker, name=spec.name, sector=sector,
            profile=profile, note="synthetic demo issuer",
        ))
    return out


def build_suite(
    issuers: Sequence[Issuer] | None = None, *, rotating_per_issuer: int = 2
) -> list[EvalCase]:
    issuers = list(issuers or UNIVERSE)
    cases: list[EvalCase] = []
    for issuer in issuers:
        cases.extend(build_issuer_cases(issuer, rotating_per_issuer=rotating_per_issuer))
    cases.extend(build_comparison_cases(issuers))
    return cases


SOURCES: dict[str, Callable[[], list[Issuer]]] = {
    "universe": lambda: list(UNIVERSE),
    "fixtures": fixture_issuers,
}


def validate_suite(session, cases: Sequence[EvalCase]) -> tuple[list[EvalCase], list[str]]:
    """Drop cases the corpus cannot score, and say why.

    A case whose issuer is absent, or whose relevance predicate matches no
    chunk, contributes a NaN to every retrieval metric and a false sense of
    coverage. Better to remove it at build time and report the removal.
    """
    from creditlens.eval.runner import relevant_chunk_ids
    from creditlens.retrieval import corpus as corpus_mod

    available = {r.ticker for r in corpus_mod.get_snapshot(session).records}
    kept: list[EvalCase] = []
    dropped: list[str] = []

    for case in cases:
        missing = [t for t in case.tickers if t.upper() not in available]
        if missing:
            dropped.append(f"{case.id}: issuer not ingested ({', '.join(missing)})")
            continue
        if case.relevance is not None and not relevant_chunk_ids(session, case.relevance):
            dropped.append(f"{case.id}: relevance predicate matched no chunk")
            continue
        kept.append(case)
    return kept, dropped


def write_suite(
    cases: Sequence[EvalCase], name: str = "universe", *, directory: Path | None = None
) -> Path:
    path = (directory or DATASET_DIR) / f"{name}.jsonl"
    payload = "\n".join(
        json.dumps(_compact(case.to_dict()), sort_keys=True) for case in cases
    )
    path.write_text(payload + "\n")
    log.info("eval suite written", extra={"suite": name, "cases": len(cases), "path": str(path)})
    return path


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items()
        if value not in (None, [], "", False)
    }
