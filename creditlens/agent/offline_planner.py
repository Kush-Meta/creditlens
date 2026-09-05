"""Deterministic planner/synthesizer used by `OfflineEngine`.

It plays the role the model plays: read the conversation so far, decide which
tools to call next, and finally write the structured analysis from the tool
results. It is rule-based and fully reproducible.

This exists for three reasons:

1. CI and unit tests exercise the whole pipeline - retrieval, calculation,
   verification, API - with no key and no network.
2. It is the fallback when the API is unreachable, so the service degrades to
   a narrower answer instead of a 500.
3. It is an eval baseline: the gap between offline and model output measures
   what the model is actually contributing, separately from what the
   deterministic layer already provides.

Its answers are labelled `engine: "offline-deterministic"` everywhere.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from creditlens.agent.providers.base import (
    AssistantTurn,
    Conversation,
    LLMResponse,
    ToolCall,
    ToolResultBatch,
    Usage,
    UserMessage,
)
from creditlens.agent.providers.offline_engine import classify_intents

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def plan_next_step(
    *, conversation: Conversation, tools: Sequence[dict[str, Any]]
) -> LLMResponse:
    state = _read_state(conversation)
    turn = state["assistant_turns"]

    if turn == 0:
        calls = _first_round(state)
    elif turn == 1:
        calls = _second_round(state)
    else:
        return LLMResponse(
            text="", stop_reason="tool_use", usage=Usage(requests=1),
            tool_calls=[ToolCall(
                id=f"offline_final_{turn}", name="submit_analysis",
                arguments=_synthesize(state),
            )],
        )
    if not calls:
        return LLMResponse(
            text="", stop_reason="tool_use", usage=Usage(requests=1),
            tool_calls=[ToolCall(id=f"offline_final_{turn}", name="submit_analysis",
                                 arguments=_synthesize(state))],
        )
    return LLMResponse(text="", tool_calls=calls, stop_reason="tool_use", usage=Usage(requests=1))


# ---------------------------------------------------------------------------
# conversation state
# ---------------------------------------------------------------------------
def _read_state(conversation: Conversation) -> dict[str, Any]:
    """Reconstruct planning state from the neutral transcript.

    Reading provider-shaped message dicts here was the last place the agent
    layer knew which vendor it was talking to; the neutral transcript removes
    that coupling and makes this markedly simpler.
    """
    question = ""
    corpus_tickers: list[str] = []
    scoped: list[str] = []
    results: list[dict[str, Any]] = []
    assistant_turns = 0

    for turn in conversation:
        if isinstance(turn, UserMessage):
            question, corpus_tickers = _absorb_user_text(turn.text, question, corpus_tickers)
            scoped.extend(_scoped_tickers(turn.text))
        elif isinstance(turn, AssistantTurn):
            assistant_turns += 1
        elif isinstance(turn, ToolResultBatch):
            for result in turn.results:
                try:
                    payload = json.loads(result.content)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    results.append(payload)

    return {
        "question": question,
        "corpus_tickers": corpus_tickers,
        "results": results,
        "assistant_turns": assistant_turns,
        # An explicit scope from the caller beats inference from the wording.
        "tickers": (
            list(dict.fromkeys(scoped)) or _target_tickers(question, corpus_tickers)
        ),
        "intents": classify_intents(question),
    }


def _absorb_user_text(
    text: str, question: str, corpus_tickers: list[str]
) -> tuple[str, list[str]]:
    """Pull the corpus roster and the question out of the preamble message.

    Both arrive in a single user message, so this parses each independently
    rather than branching on which one the message "is".
    """
    for line in text.splitlines():
        match = re.match(r"^- ([A-Z0-9\.\-]{1,6}) \(([^;)]+)", line)
        if match:
            # Store "TICKER::Company Name" so name mentions resolve too.
            entry = f"{match.group(1)}::{match.group(2).strip()}"
            if entry not in corpus_tickers:
                corpus_tickers.append(entry)
    if "QUESTION:" in text:
        return text.split("QUESTION:", 1)[1].strip(), corpus_tickers
    if question:
        return question, corpus_tickers
    if text.startswith("Issuers currently in the corpus"):
        return question, corpus_tickers
    return text.strip(), corpus_tickers


_SCOPE_RE = re.compile(r"user scoped this question to:\s*([A-Z0-9\.\-, ]+)", re.I)


def _scoped_tickers(text: str) -> list[str]:
    """Read the caller-supplied issuer scope out of the preamble."""
    match = _SCOPE_RE.search(text or "")
    if not match:
        return []
    return [
        token for token in
        (t.strip().strip(".").upper() for t in match.group(1).split(","))
        if token
    ]


def _target_tickers(question: str, corpus: list[str]) -> list[str]:
    """Resolve issuers mentioned by ticker or by company name."""
    upper = question.upper()
    found: list[str] = []
    for entry in corpus:
        ticker, _, name = entry.partition("::")
        if re.search(rf"\b{re.escape(ticker)}\b", upper):
            found.append(ticker)
            continue
        # Match the distinctive first word of the company name, by prefix:
        # "Kohl's" in a question must resolve to the filer name "KOHLS CORP",
        # and the apostrophe splits the token.
        head = re.sub(r"[^A-Z ]", "", name.upper()).split()
        if head and len(head[0]) > 3:
            stem = head[0][:5]
            if any(
                token.startswith(stem) or head[0].startswith(token)
                for token in re.findall(r"[A-Z]{4,}", upper)
            ):
                found.append(ticker)
    if found:
        return found
    # No corpus issuer is named. If the question clearly names *some* company,
    # return nothing so the caller reports the gap instead of silently
    # analysing whichever issuer happens to be first.
    if _names_unknown_entity(question, corpus):
        return []
    return [entry.partition("::")[0] for entry in corpus[:1]]


_ENTITY_STOPWORDS = {
    "What", "How", "Has", "Is", "Are", "Does", "Do", "Compare", "Calculate",
    "Summarize", "Which", "Why", "When", "Where", "Explain", "Show", "Give",
    "The", "A", "An", "Item", "Risk", "Factors", "EBITDA", "FCF", "TTM",
}


def _names_unknown_entity(question: str, corpus: list[str]) -> bool:
    """True when the question names a proper noun that is not in the corpus."""
    known = set()
    for entry in corpus:
        ticker, _, name = entry.partition("::")
        known.add(ticker.upper())
        known.update(re.sub(r"[^A-Za-z ]", "", name).upper().split())
    candidates = [
        token for token in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", question)
        if token not in _ENTITY_STOPWORDS
    ]
    return any(token.upper() not in known for token in candidates)


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def _first_round(state: dict[str, Any]) -> list[ToolCall]:
    tickers = state["tickers"]
    if not tickers:
        return [ToolCall("offline_list", "list_companies", {})]

    intents = state["intents"]
    calls: list[ToolCall] = []
    primary = tickers[0]

    calls.append(ToolCall("offline_fin", "get_financials",
                          {"ticker": primary, "freq": "quarterly", "periods": 6}))
    calls.append(ToolCall("offline_ratios", "compute_ratios",
                          {"ticker": primary, "period": "TTM"}))

    if "trend" in intents or "profile" in intents:
        for i, metric in enumerate(("net_debt_to_ebitda", "ebitda_interest_coverage",
                                    "operating_margin", "fcf_margin")):
            calls.append(ToolCall(f"offline_trend_{i}", "metric_trend",
                                  {"ticker": primary, "metric": metric,
                                   "freq": "quarterly", "lookback": 6}))
    if "cause" in intents:
        calls.append(ToolCall("offline_cmp", "compare_periods",
                              {"ticker": primary, "metric": "operating_margin"}))
    query = state["question"] or "liquidity leverage risk factors"
    calls.append(ToolCall("offline_search", "search_filings",
                          {"query": query, "tickers": tickers, "top_k": 6}))
    return calls


def _second_round(state: dict[str, Any]) -> list[ToolCall]:
    tickers = state["tickers"]
    intents = state["intents"]
    calls: list[ToolCall] = []
    if not tickers:
        # list_companies has run and nothing matched: stop and report the gap
        # rather than analysing an arbitrary issuer.
        return []
    if len(tickers) >= 2 or "compare" in intents:
        calls.append(ToolCall("offline_peer", "compare_companies",
                              {"tickers": tickers[:4], "period": "TTM"}))
    calls.append(ToolCall("offline_score", "credit_scorecard",
                          {"ticker": tickers[0], "period": "TTM"}))
    # If the ratio engine reported anything as uncomputable, find out exactly
    # what is missing so the answer can name the gap instead of ignoring it.
    if any(block.get("unavailable") for block in _find_all(state, "ratios")):
        calls.append(ToolCall("offline_cov", "data_coverage", {"ticker": tickers[0]}))
    if "risk" in intents:
        calls.append(ToolCall("offline_risk", "search_filings", {
            "query": state["question"] or "principal risk factors",
            "tickers": tickers, "items": ["1A"], "top_k": 5,
        }))
    return calls


def _find_result(state: dict[str, Any], key: str) -> dict[str, Any] | None:
    for result in state["results"]:
        if isinstance(result, dict) and key in result:
            return result
    return None


def _find_all(state: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [r for r in state["results"] if isinstance(r, dict) and key in r]


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
_DIRECTION_WEIGHTS = {
    "net_debt_to_ebitda": 1.4,
    "debt_to_ebitda": 1.2,
    "ebitda_interest_coverage": 1.2,
    "operating_margin": 1.0,
    "fcf_margin": 1.0,
    "current_ratio": 0.6,
}


def _synthesize(state: dict[str, Any]) -> dict[str, Any]:
    trends = [r for r in _find_all(state, "series") if r.get("direction")]
    ratio_blocks = _find_all(state, "ratios")
    scorecards = [r for r in state["results"] if isinstance(r, dict) and "composite_score_0_100" in r]
    searches = _find_all(state, "passages")
    comparison = _find_result(state, "comparison")
    financials = _find_result(state, "revenue_growth")

    direction, direction_detail = _direction(trends)
    key_metrics = _key_metrics(ratio_blocks, trends)
    positives, risks = _factors(scorecards, searches)
    citations = sorted(
        {p["citation"] for search in searches for p in search.get("passages", [])},
        key=lambda label: int(label[1:]) if label[1:].isdigit() else 0,
    )

    scope = state["tickers"] or []
    if not scope:
        return _unavailable_payload(state)
    ticker = " vs ".join(scope) if len(scope) > 1 else scope[0]
    answer = _answer_text(ticker, direction, key_metrics, scorecards, citations)
    reasoning = _reasoning_text(
        ticker, direction_detail, key_metrics, financials, comparison, scorecards, searches
    )

    caveats = [
        "Produced by the deterministic offline engine (no language model was "
        "available); the narrative is templated from tool output rather than "
        "written by a model.",
    ]
    coverage = _find_result(state, "concepts_missing")
    if coverage and coverage.get("concepts_missing"):
        caveats.append(
            f"{coverage['ticker']} does not report "
            f"{', '.join(coverage['concepts_missing'])} in its XBRL facts, so "
            f"{len(coverage.get('ratios_blocked', []))} ratio(s) including "
            "leverage measures could not be computed."
        )
    for card in scorecards:
        caveats.extend(card.get("warnings", [])[:2])
    if any(block.get("is_synthetic") for block in ratio_blocks):
        caveats.append("Figures come from synthetic demo data, not filed results.")

    return {
        "answer": answer,
        "credit_direction": direction,
        "key_metrics": key_metrics,
        "positive_factors": positives,
        "risk_factors": risks,
        "reasoning": reasoning,
        "citations": citations,
        "confidence": "medium" if key_metrics and citations else "low",
        "caveats": caveats[:5],
    }


def _unavailable_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Answer for a question about an issuer that has not been ingested."""
    listing = _find_result(state, "companies") or {}
    available = [c["ticker"] for c in listing.get("companies", [])]
    named = ", ".join(
        sorted({
            token for token in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", state["question"])
            if token not in _ENTITY_STOPWORDS
        })
    )
    return {
        "answer": (
            f"No issuer matching {named or 'the company named'} has been ingested, so "
            "there is no filing or XBRL data to analyse. The corpus currently holds: "
            f"{', '.join(available) or 'no issuers'}. Ingest the company first "
            "(POST /api/ingest) and re-run the question."
        ),
        "credit_direction": "not_applicable",
        "key_metrics": [],
        "positive_factors": [],
        "risk_factors": [],
        "reasoning": (
            f"The question names {named or 'a company'}, which is not present in the "
            "corpus. Answering from general knowledge would produce figures that "
            "cannot be traced to any filing, so no analysis is offered."
        ),
        "citations": [],
        "confidence": "low",
        "caveats": [
            "Issuer not present in the corpus; no analysis attempted.",
            "Produced by the deterministic offline engine.",
        ],
    }


def _direction(trends: list[dict[str, Any]]) -> tuple[str, list[str]]:
    score = 0.0
    detail: list[str] = []
    for trend in trends:
        metric = trend.get("metric")
        weight = _DIRECTION_WEIGHTS.get(metric, 0.5)
        heading = trend.get("direction")
        if heading == "improving":
            score += weight
        elif heading == "deteriorating":
            score -= weight
        if heading in {"improving", "deteriorating", "stable"} and trend.get("series"):
            first = next((p for p in trend["series"] if p.get("value") is not None), None)
            last = next((p for p in reversed(trend["series"]) if p.get("value") is not None), None)
            if first and last:
                detail.append(
                    f"{trend.get('label', metric)} moved from {first['display'] if 'display' in first else first['value']}"
                    f" in {first['period']} to {last['display'] if 'display' in last else last['value']}"
                    f" in {last['period']} ({heading})"
                )
    if not trends:
        return "not_applicable", detail
    if score >= 1.0:
        return "improving", detail
    if score <= -1.0:
        return "deteriorating", detail
    if abs(score) < 0.5:
        return "stable", detail
    return "mixed", detail


_HEADLINE_RATIOS = (
    "net_debt_to_ebitda", "debt_to_ebitda", "ebitda_interest_coverage",
    "current_ratio", "cash_to_debt", "ebitda_margin", "operating_margin",
    "fcf_margin", "cfo_to_debt",
)


def _key_metrics(
    ratio_blocks: list[dict[str, Any]], trends: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    trend_by_metric = {t.get("metric"): t for t in trends}
    for block in ratio_blocks:
        for ratio in block.get("ratios", []):
            if ratio["name"] not in _HEADLINE_RATIOS or ratio.get("value") is None:
                continue
            key = (ratio["ticker"], ratio["period"], ratio["name"])
            if key in seen:
                continue
            seen.add(key)
            trend = trend_by_metric.get(ratio["name"])
            commentary = ratio.get("formula", "")
            if trend and trend.get("percent_change") is not None:
                commentary = (
                    f"{trend['direction']}; {trend['percent_change']:+.1f}% across the "
                    f"observed window. Formula: {ratio.get('formula')}"
                )
            out.append({
                "name": ratio["label"],
                "value": ratio["display"],
                "period": ratio["period"],
                "ticker": ratio["ticker"],
                "commentary": commentary,
            })
    return out[:10]


def _factors(
    scorecards: list[dict[str, Any]], searches: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    positives: list[str] = []
    risks: list[str] = []
    for card in scorecards:
        positives.extend(card.get("strengths", [])[:3])
        risks.extend(card.get("weaknesses", [])[:3])
    for search in searches:
        for passage in search.get("passages", [])[:3]:
            snippet = " ".join((passage.get("text") or "").split())[:220]
            if not snippet:
                continue
            entry = f"{snippet}... [{passage['citation']}]"
            if passage.get("item") == "1A":
                risks.append(entry)
            else:
                positives.append(entry)
    return positives[:5], risks[:6]


def _answer_text(
    ticker: str,
    direction: str,
    key_metrics: list[dict[str, Any]],
    scorecards: list[dict[str, Any]],
    citations: list[str],
) -> str:
    headline = ", ".join(f"{m['name']} {m['value']}" for m in key_metrics[:3]) or "no computable ratios"
    parts = [
        f"Credit direction for {ticker} reads as {direction.replace('_', ' ')} on the "
        f"available periods, driven by {headline}."
    ]
    if scorecards:
        card = scorecards[0]
        if card.get("composite_score_0_100") is not None:
            parts.append(
                f"The internal scorecard (a heuristic model, not a rating) puts the "
                f"composite at {card['composite_score_0_100']}/100, an implied "
                f"{card['implied_band']} band, at {card['confidence']} factor coverage."
            )
    if citations:
        parts.append(f"Supporting filing passages: {', '.join(citations[:6])}.")
    return " ".join(parts)


def _reasoning_text(
    ticker: str,
    direction_detail: list[str],
    key_metrics: list[dict[str, Any]],
    financials: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    scorecards: list[dict[str, Any]],
    searches: list[dict[str, Any]],
) -> str:
    paragraphs: list[str] = []
    if direction_detail:
        paragraphs.append("Trend evidence: " + "; ".join(direction_detail[:4]) + ".")
    if financials and financials.get("revenue_growth"):
        moves = ", ".join(
            f"{g['from']}->{g['to']} {g['percent_change']:+.1f}%"
            for g in financials["revenue_growth"][-4:]
        )
        paragraphs.append(f"Sequential revenue growth for {ticker}: {moves}.")
    if key_metrics:
        paragraphs.append(
            "Current levels: "
            + "; ".join(f"{m['name']} {m['value']} ({m['period']})" for m in key_metrics[:6])
            + "."
        )
    if comparison:
        winners = comparison.get("stronger_on_each_ratio", {})
        stated = ", ".join(f"{k}: {v}" for k, v in list(winners.items())[:6] if v)
        if stated:
            paragraphs.append(f"On a like-for-like comparison the stronger issuer per ratio is - {stated}.")
    if scorecards:
        card = scorecards[0]
        drivers = sorted(
            [f for f in card.get("factors", []) if f.get("score_0_100") is not None],
            key=lambda f: f["score_0_100"],
        )
        if drivers:
            weakest = drivers[0]
            strongest = drivers[-1]
            paragraphs.append(
                f"Scorecard drivers: weakest factor is {weakest['label']} at "
                f"{weakest['display']} ({weakest['score_0_100']}/100); strongest is "
                f"{strongest['label']} at {strongest['display']} "
                f"({strongest['score_0_100']}/100)."
            )
    cited = [
        f"[{p['citation']}] {p.get('source', '')} item {p.get('item') or 'n/a'}"
        for search in searches for p in search.get("passages", [])[:4]
    ]
    if cited:
        paragraphs.append("Filing evidence reviewed: " + "; ".join(cited[:6]) + ".")
    return "\n\n".join(paragraphs) or "Insufficient tool output to construct a reasoning chain."
