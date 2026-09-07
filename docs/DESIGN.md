# CreditLens — Design Document

**Status:** working system, measured. Version 0.1.0.
**Scope:** what was built, why each decision was made, what was measured, what
was tried and abandoned, and what remains unknown.

This document is written for someone who has to maintain or extend the system.
It records reasoning and evidence, not just structure — including the places
where a hypothesis was tested and turned out to be wrong.

---

## Contents

1. [Problem and thesis](#1-problem-and-thesis)
2. [System overview](#2-system-overview)
3. [Data layer](#3-data-layer)
4. [Deterministic calculation engine](#4-deterministic-calculation-engine)
5. [Retrieval](#5-retrieval)
6. [Agent](#6-agent)
7. [Provider abstraction](#7-provider-abstraction)
8. [Verification](#8-verification)
8b. [Provenance surface](#8b-provenance-surface)
9. [Evaluation](#9-evaluation)
10. [Scaling](#10-scaling)
11. [Embedding experiment: a negative result](#11-embedding-experiment-a-negative-result)
12. [Observability](#12-observability)
13. [Testing strategy](#13-testing-strategy)
14. [Decisions and rejected alternatives](#14-decisions-and-rejected-alternatives)
15. [Known limitations](#15-known-limitations)
16. [Roadmap](#16-roadmap)

---

## 1. Problem and thesis

### The problem

A credit analyst answering *"has Oracle's leverage improved?"* does four things:
pulls the right financial data, computes the right ratios over the right
periods, finds what management said about the drivers, and forms a view that
distinguishes structural change from cyclical noise. Each step has a different
failure mode, and only the last is genuinely a language task.

A "chat with a 10-K" system collapses all four into one language task. That
fails in a specific and dangerous way: language models produce *plausible*
numbers. A leverage ratio that is wrong by 30% reads exactly like one that is
right, and in credit analysis the number is the conclusion.

### The thesis

> **The language model reasons and explains. Ordinary code computes.**

Concretely, three rules:

1. **No figure in an answer originates from the model.** Every number comes
   from a deterministic calculation engine.
2. **Every number is registered before use.** Tool output enters an evidence
   ledger before it re-enters the conversation.
3. **The answer is checked against the ledger after it is written.** Numbers
   that cannot be traced are reported as unsupported rather than shipped.

Rule 3 is what makes the thesis testable rather than aspirational. Prompting a
model to "only use numbers from tools" is a request; checking afterwards is a
measurement, and it is the metric the evaluation suite gates on.

### Non-goals

- Not a trading system, not investment advice, not a credit rating.
- Not a general document Q&A system. The tool surface is credit-specific by
  design; that is what makes tool selection tractable.
- Not a multi-agent framework. See §14.

---

## 2. System overview

```
SEC EDGAR ──▶ INGESTION ──▶ STORE ──┬──▶ FINANCE (deterministic) ──┐
                                     └──▶ RETRIEVAL (hybrid)  ──────┤
                                                                    ▼
                                                          AGENT (one loop)
                                                                    │
                                                     evidence ledger│
                                                                    ▼
                                                            VERIFIER
                                                                    │
                                                  ┌─────────────────┼──────────┐
                                                  ▼                 ▼          ▼
                                              FastAPI            CLI      Evaluation
```

**Layer boundaries and why they are where they are:**

| Layer | Responsibility | Knows nothing about |
|---|---|---|
| Ingestion | EDGAR → normalized facts + item-tagged chunks | ratios, retrieval, models |
| Finance | facts → ratios, trends, attribution, scorecard | retrieval, models, HTTP |
| Retrieval | query → ranked passages with citations | finance, models |
| Agent | orchestrate tools, produce structured output | which vendor answers |
| Verifier | check the answer against the ledger | how the answer was produced |
| Surfaces | HTTP, CLI, evaluation | each other |

The key property: **every deterministic layer is usable and testable without an
LLM.** That is why 263 tests run with no key and no network, and why the
evaluation harness can measure retrieval and calculation independently of any
model.

**Module map**

```
creditlens/
├── config.py                  settings + a fingerprint of answer-affecting knobs
├── db/                        SQLAlchemy models, session, additive migrations
├── ingest/
│   ├── edgar_client.py        rate limit, disk cache, retry
│   ├── xbrl_normalizer.py     fiscal calendars, dedup, YTD differencing, Q4 rebuild
│   ├── filing_parser.py       HTML → item-tagged sections
│   ├── chunker.py             paragraph-aware chunking with overlap
│   ├── universe.py            the curated 42-issuer universe
│   ├── pipeline.py            fetch (parallel) / persist (serial) / batch ingest
│   └── fixtures.py            synthetic demo corpus
├── finance/
│   ├── taxonomy.py            XBRL tags → canonical concepts
│   ├── statements.py          period assembly, TTM, derived concepts
│   ├── ratios.py              29 ratios as declared objects
│   ├── trends.py              exact change attribution
│   └── scorecard.py           weighted factors + Altman Z''
├── retrieval/
│   ├── embeddings.py          Embedder ABC + hashed/ollama/openai/google
│   ├── bm25.py                numpy-backed inverted index
│   ├── corpus.py              in-memory snapshot with vectorised filters
│   ├── hybrid.py              RRF fusion, section priors, issuer quota, MMR
│   └── reembed.py             embedding-model migration
├── agent/
│   ├── providers/             base + anthropic + openai + google + offline
│   ├── tools.py               10 tools over deterministic code
│   ├── evidence.py            the evidence ledger
│   ├── orchestrator.py        the controlled loop
│   ├── verifier.py            numeric grounding + citation validity
│   └── offline_planner.py     the deterministic engine's planner
├── eval/                      dataset, generator, metrics, runner
├── api/                       FastAPI
└── observability/             logging, tracing, metrics
```

---

## 3. Data layer

### 3.1 Sources

Two, joined through `Filing` so a narrative claim and a number can be traced to
the same document:

- **XBRL `companyfacts`** — every numeric fact a filer has tagged.
- **10-K / 10-Q documents** — the narrative, parsed into item-tagged sections.

### 3.2 What real filings do to a naive pipeline

Each of the following broke the pipeline against live SEC data. Each is now a
fixed behaviour with a regression test in `tests/test_ingest.py`.

**(1) `fy` describes the filing, not the fact.** Every 10-K restates two prior
years as comparatives, and all three carry the *filing's* fiscal year. Because
the dedup key includes fiscal year, trusting `fy` files FY2023 revenue under
FY2025 and silently mixes periods inside one "year".

*Fix:* period labels derive from the period end date, via a fiscal calendar
inferred per company — modal year-end month, plus a year offset calibrated from
annual facts whose `fy` is trustworthy (the filing's own primary period). One
mechanism handles Oracle (FY ends May 2026 = FY2026), Apple (late September),
Microsoft (June) and retail calendars (FY ending January 2025 = fiscal 2024).

**(2) Fiscal-year rollover.** A quarter ending September belongs to FY2026 for a
June-year-end filer, not FY2025. Getting this wrong produced non-contiguous
quarter series for every non-calendar filer.

**(3) 52/53-week drift.** Apple's year end can land on 1 October. Quarter
assignment measures distance to the year end in days and snaps to the nearest
quarter, with a ten-day tolerance on the rollover.

**(4) 10-Q cash flows are cumulative.** The Q3 filing reports nine months, not
three. Without differencing (`Q2 = YTD6 − Q1`, `Q3 = YTD9 − YTD6`) quarterly
operating cash flow, capex and D&A do not exist — which removes EBITDA, free
cash flow and every ratio built on them from the TTM view.

**(5) No 10-Q for Q4.** `Q4 = FY − (Q1+Q2+Q3)`, flow concepts only, marked
`derived-q4` so nothing mistakes it for a reported figure.

**(6) Year-end balances belong to both Q4 and FY.** Otherwise annual ratio views
have income-statement data and no balance sheet.

**(7) Tag variety.** Microsoft and Oracle never emit a combined D&A tag; they
report `Depreciation` and `AmortizationOfIntangibleAssets` separately. A
canonical taxonomy with priority-ordered aliases resolves this, and the losing
tag is retained in `raw_concept` for audit.

**(8) `LongTermDebt` sometimes includes current maturities.** Adding short-term
debt on top double-counts. Total debt has three paths in trust order: a
filer-reported combined total, components summed, or an inclusive long-term tag
used alone — each recording which path it took.

**(9) Filing HTML fights back.** Every item heading appears in the table of
contents before it appears as a heading, and Microsoft prints "Item 7" atop
*every page* of the MD&A (fifteen matches inside one section).

**(10) Contents blocks are not always at the front, and item numbers are not
always repeated.** GE and Honeywell publish their item cross-index at the *back*
of the 10-K. Honeywell renders every heading with its page number in the same
table row (`30 | ITEM 1A. | Risk Factors`). Chevron, Exxon and GE introduce MD&A
by title alone.

### 3.3 Section detection algorithm

Arrived at empirically; the order matters.

1. Match item headings, tolerating table-cell separators, `PART II` labels and
   page-number prefixes.
2. **Remove the contents block** — a dense run of matches spanning ≥6 distinct
   items — wherever it sits in the document.
3. **Remove closely-spaced individual lines**, which catches contents lists too
   small to register as a block. An item appearing exactly once is exempt: a
   filer rendering `ITEM 7.` alone on a line produces a short gap but is still
   the real heading.
4. If fewer than three headings survive, **fall back to title matching**
   (`RISK FACTORS`, `MANAGEMENT'S DISCUSSION AND ANALYSIS`, ...), excluding the
   contents span. Title-derived headings skip the canonical-order filter,
   because filers that use them also tend to present sections out of order.
5. Take the **first** surviving occurrence per item (later ones are page
   headers), end each section at the **next selected section**, and keep the
   **longest canonically-ordered subsequence** so one stray cross-reference
   cannot truncate the document.
6. Split the audited statements out as Item 8 wherever they landed.

> **Why title matching is a fallback and not a peer.** It was first implemented
> as an equal source of headings. That made good parses *worse*: the title
> patterns fire on cross-references and contents rows, and Pfizer went from
> parsing correctly to producing nothing. Demoting it to a fallback fixed GE and
> Honeywell without disturbing the 40 issuers that already worked.

**Result:** 42 of 46 issuers (91%) yield both Item 1A and Item 7. The four that
do not still have their text ingested and retrievable; only the item label is
missing.

### 3.4 The issuer universe

42 real issuers selected for **credit dispersion, not market capitalisation**. A
corpus of mega-cap technology names is broad but analytically flat: leverage and
coverage cluster in a narrow band, so neither the ratio engine nor the scorecard
can be seen to differentiate.

| Sampling profile | n | Contributes |
|---|---|---|
| `net_cash` | 7 | cash exceeds debt |
| `investment_grade` | 12 | moderate leverage, comfortable coverage |
| `leveraged` | 11 | debt-funded M&A or capital return |
| `structurally_levered` | 6 | utilities and REITs — high leverage is normal |
| `cyclical_stressed` | 6 | airlines, cruise, weak retail |

`profile` is a **sampling label recording why an issuer is in the universe** —
never a rating. The credit view is computed from each issuer's filings; comparing
the two is a sanity check on the scorecard, not a target to fit it to.

### 3.5 Ingestion concurrency

Network fetch fans out across a thread pool; persistence stays on one thread.
EDGAR is dominated by download latency, SQLite tolerates exactly one writer, and
the rate limiter is shared so parallelism never breaches the SEC's budget. Runs
are resumable. **42 issuers, 516 filings, 5 minutes, zero failures.**

### 3.6 Schema and migrations

`create_all` creates missing tables but never alters existing ones, so a column
added after a database exists is silently absent. `db/migrations.py` applies
additive column migrations idempotently at startup — deliberately additive only:
no drops, no type changes. Anything beyond that belongs in Alembic.

---

## 4. Deterministic calculation engine

### 4.1 Ratios as declared objects

29 ratios, each a `RatioDef` with inputs, formula, polarity and guards — not an
inline expression. Three properties follow:

- **Explainability.** Every result carries its formula, the exact inputs it
  consumed, and the provenance of each input.
- **Guardrails.** A ratio with a meaningless denominator returns `None` *plus a
  reason* — `"ebitda is -412,000,000; the ratio is not economically meaningful
  with a non-positive denominator"` — rather than a number that looks
  authoritative.
- **Introspection.** The agent lists available ratios instead of inventing
  formulas.

Stock/flow confusion is impossible by construction: `taxonomy.is_flow` gates
every aggregation, so a balance sheet can never be summed across quarters.

### 4.2 Change attribution is arithmetic

For any ratio N/D the change decomposes exactly:

```
N₁/D₁ − N₀/D₀  =  (N₁−N₀)/D₁  −  N₀(D₁−D₀)/(D₀D₁)
                  └ numerator ┘   └──── denominator ────┘
```

So *"margin fell 180 bps: −240 bps from cost growth, +60 bps from revenue
growth"* is computed, and the components provably sum to the total (asserted in
tests). Margins get a second line-by-line bridge across expense lines as a share
of revenue, with an explicit residual for items the filer did not tag
separately.

### 4.3 Scorecard

Ten weighted factors with piecewise-linear anchor curves → 0–100 composite →
implied band, plus an Altman Z''-score. Weights renormalise over computable
factors and the covered share is reported, so a score built from 40% of the
factors never looks as confident as one built from all of them.

The anchors are judgement, not a fitted model. With no labelled default data in
the repository, a fitted model would be false precision; anchors are auditable
line by line. The output is labelled a heuristic in every response containing it.

---

## 5. Retrieval

Hybrid BM25 + dense, fused with Reciprocal Rank Fusion.

1. **Metadata pre-filter** — ticker, form, fiscal year, item. Analyst questions
   are scoped, and filtering before scoring is faster and more precise than
   hoping the ranker infers scope.
2. **Query expansion** — a curated credit vocabulary (`leverage` → borrowings,
   indebtedness). Additive, lexical leg only, so a bad expansion cannot hijack
   the dense ranking.
3. **RRF** — fuses by *rank*, not score, so BM25's unbounded scores and cosine's
   [−1,1] never need calibrating. That is the classic failure of naive
   score-weighted hybrid search.
4. **Section priors** — a risk question prefers Item 1A. Small multiplicative
   priors, never enough to override a strong direct match.
5. **Per-issuer quota** — a comparison question is useless if all eight passages
   come from one issuer, which is what a global ranking produces when one company
   phrases the topic more strongly.

**Measured defaults** (`scripts/sweep_retrieval.py`, 24 configurations):

- **MMR off.** Costs 0.10–0.24 nDCG@8. Relevant passages in filings genuinely
  cluster — four quarters of the same liquidity discussion are all relevant — so
  diversity trades away hits.
- **`dense_weight = 0.2`.** Best on the paraphrase probe at negligible cost on
  keyword queries.

---

## 6. Agent

**One agent, one loop, ten tools, hard budgets.** No planner/critic hierarchy,
no dynamic agent spawning — for a bounded analytical task those add latency and
failure modes without improving answers, and they make the trace much harder to
reason about.

```
list_companies · get_financials · compute_ratios · compare_periods · metric_trend
credit_scorecard · compare_companies · search_filings · data_coverage
                              ↓
                        submit_analysis   (structured final answer, exactly once)
```

**Invariants the loop enforces regardless of model behaviour:**

- Every tool result is deterministic code output, registered in the evidence
  ledger before it re-enters the conversation.
- The loop always terminates — iteration cap, tool-call cap, forced final turn.
- An answer is always produced: provider failure degrades to the deterministic
  engine *mid-run* and records why.
- A failed tool call returns `is_error` **plus the valid alternatives**, so the
  model can recover rather than the turn dying.

The final answer is a forced tool call rather than parsed prose, which makes the
output schema a contract instead of a hope.

### The offline engine

A deterministic planner/synthesizer that runs the same loop and calls the same
finance code, writing its narrative from tool results. It is not a mock. It
serves three purposes: CI and tests run with no key or network; the service
degrades to it rather than erroring; and it is the eval baseline against which a
model's contribution can be measured.

---

## 7. Provider abstraction

The agent is vendor-neutral. Fourteen providers ship: Anthropic (direct,
Bedrock, Vertex), the OpenAI protocol (OpenAI, Azure, Groq, Together,
OpenRouter, DeepSeek, Mistral), Google Gemini, self-hosted (Ollama, vLLM), and
the offline engine.

### The hard part was the conversation format

The `LLMEngine` seam already existed. What was coupled was the message shape,
and the three tool protocols disagree irreconcilably:

| | Assistant's tool call | Result returns as |
|---|---|---|
| Anthropic | `tool_use` block inside content | `tool_result` blocks in one **user** message |
| OpenAI | `tool_calls` on the message | one message per result, `role="tool"` + `tool_call_id` |
| Google | `functionCall` part | `functionResponse` parts under `role="user"`, correlated **by function name** — no call id exists |

**Design:** the loop speaks in neutral `Turn` objects — `UserMessage`,
`AssistantTurn`, `ToolResultBatch` — and each engine translates the whole
transcript into its wire format on every call. Engines are **stateless with
respect to the conversation**, which makes retries, replay and *substituting the
engine mid-run* trivially correct. That last property is how a provider outage
degrades inside an analysis rather than failing the request.

`AssistantTurn` carries the provider's own representation, replayed verbatim
where required — Anthropic thinking blocks must be echoed back unchanged, and
reconstructing them from text would silently drop them.

### Details real providers forced

- Google's schema dialect rejects `additionalProperties`; schemas are sanitised.
- OpenAI reports cached tokens *inside* `prompt_tokens`; billing both charges the
  same tokens twice at two rates.
- Reasoning models take `max_completion_tokens`.
- Weaker local models emit malformed tool-argument JSON — surfaced as a
  recoverable tool error, not a crash.
- Keyless local servers cannot be detected by "is a key set". They are probed by
  requesting the model list and requiring a JSON answer; a bare TCP connect
  reported an unrelated web app on port 8000 as a working model server.

### Cost accounting without inventing numbers

Anthropic rates ship built in. Every other model reports
`cost_basis: "no rate configured"` rather than `$0.00`, configurable through
`CREDITLENS_MODEL_PRICES`. In a system whose premise is refusing to state
unsupported numbers, silently inventing a price would be self-defeating. Local
providers are priced at zero because that is true, not assumed.

---

## 8. Verification

Every figure in the answer is extracted with its unit and classified:

| Status | Meaning |
|---|---|
| `verified` | matches a ledger value within tolerance |
| `cited` | quoted verbatim from a retrieved passage |
| `contradicted` | near a computed value but outside tolerance — the most serious class, because it looks plausible |
| `unsupported` | traceable to nothing |

Three details make it work rather than look like it works. Each was found by a
false positive, not by design:

1. **Display precision sets a floor on tolerance.** `"18.6%"` asserts 18.6 ±
   0.05, not 18.600000. Without this, a value computed as 18.63 and faithfully
   rounded is scored a contradiction. Fixing it moved numeric accuracy from 0.62
   to 1.00 with no change to the generator. A floating-point epsilon is also
   required: `|1.1 − 1.05|` evaluates to `0.05000000000000004`.
2. **Matching is metric-aware, scoped to the clause.** Candidates are restricted
   to evidence whose own clause names them, using label tokens plus a phrase
   vocabulary (`"gross leverage"` → debt/EBITDA). Without it, *"the current ratio
   was 1.29x"* matches a debt/equity of 1.29 and scores verified. Sentence scope
   proved too coarse for prose listing several metrics in one sentence, so the
   scope is the comma-delimited clause, widening to the sentence only when the
   clause names nothing.
3. **Quotations outrank fuzzy proximity.** A figure quoted from a cited filing is
   grounded by the citation even when it sits within 25% of some computed value.
   Grounding requires the numeric token *and* surrounding wording to survive
   verbatim, checked on both sides independently.

Confidence is derived from these rates plus evidence breadth, and is **capped by,
not taken from,** the model's stated confidence.

**Adversarial suite: 10/10**, including a fabricated `$8.4 billion of senior
notes` written in the same phrasing as a real quotation, and a real number
attached to the wrong metric.

---

## 8b. Provenance surface

A second, deliberately non-LLM surface: a chat that answers *where did this
number come from*.

**Why it is not an agent.** Analysis is a language task; lineage is not. The
answer is already recorded exactly, in the provenance the calculation engine
attaches to every value. A model asked to infer it would be slower, more
expensive, and capable of being wrong about the one thing the surface exists to
be right about. So `agent/provenance.py` is deterministic end to end: intent is
matched by pattern, and every fact in the reply is read from the store.

**The resolver** walks recorded provenance recursively:

- `derived-ttm` → recurse into the same concept across the window's quarters
- `derived` → recurse into the sibling concepts named in `derived_from`
- otherwise → terminate at the stored fact, resolving its accession to a filing
  and a link on sec.gov

Depth-bounded, and where no fact row matches the concept name (a value mirrored
from a differently-named tag, such as `total_debt` from a filer-reported
combined amount) it falls back to the accession and tag carried on the
provenance itself.

**The invariant, asserted in tests:** every leaf carries an accession, a formula,
or an explicit origin. Nothing appears from nowhere.

**Two latent bugs this surface exposed:**

1. Filings were labelled by *calendar* quarter. Oracle's 10-Q ending 28 February
   is fiscal Q3 and displayed as Q1. Wrong wherever a filing is shown, and
   fatal for a lineage view. Fixed in `_fiscal_from_ref` by threading the
   inferred fiscal calendar through ingestion, plus a backfill
   (`relabel_filing_periods`) that corrects existing corpora from the fact table
   without re-downloading anything — 96 filings relabelled.
2. Static assets were cache-stale: a browser reusing an old `app.js` against new
   HTML fails in a way indistinguishable from a code bug. `Cache-Control:
   no-cache` alone does not rescue an already-stored entry, so asset URLs are
   content-hashed.

---

## 9. Evaluation

209 cases across three suites:

- **`golden`** (26) — hand-written, covering cases needing judgement: causal
  attribution, data-gap honesty, an issuer deliberately absent.
- **`universe`** (167) — generated per issuer, so coverage tracks the corpus.
- **`fixtures`** (16) — generated from the synthetic issuers so CI can gate
  without a network.

Generation is deterministic, gives every issuer two core families plus a rotating
pair, and adds one within-sector comparison per sector — paired across
*differing* credit profiles so the comparison discriminates. Three rules keep
generated cases from becoming filler: every case asserts something checkable,
generation is reproducible, and **cases the corpus cannot score are dropped at
build time**.

**Methodology and its limits.** Relevance labels are *programmatic*: a case
declares issuer, item and required terms, and the harness resolves that against
the corpus. Recall is well-defined and labels stay consistent as the corpus
grows. But keyword-defined labels cannot be assumed neutral between lexical and
semantic retrieval — see §11, where this became the binding constraint.

Numeric ground truth is **recomputed independently** through the finance layer,
not read from the agent's tool output; checking the agent against its own tools
would only prove it can copy. Metrics an issuer genuinely cannot support are
excluded from scoring, so an honest "not available" is never penalised.

### Measured results (deterministic engine, 193 cases, 46 issuers)

| Metric | Value |
|---|---|
| Numeric accuracy | 1.000 |
| Unsupported claim rate | 0.000 |
| Citation validity | 1.000 |
| Ground-truth metric accuracy | 0.927 |
| Retrieval nDCG@8 | 0.574 |
| Credit-direction accuracy | 1.000 |
| Tool-selection F1 | 0.368 |
| Retrieval latency p50 / p95 | 31 / 44 ms |

> **These measure the deterministic path only.** No model has been run through
> the loop against a live provider. Numeric accuracy of 1.000 is measured against
> a generator that *cannot* fabricate by construction, so it is a floor on the
> verifier's false-positive rate, not evidence about a model. Tool-selection F1
> is depressed for the same reason — the deterministic planner follows a fixed
> plan.

---

## 10. Scaling

The universe expansion took the corpus from 3,609 to 76,758 chunks. Three things
that were free at the smaller size were not.

**Memory: 1.99 GB → 484 MB.** Tokenised text retained on every chunk record cost
~844 MB; it is now built during index construction and discarded. BM25 postings
as `dict[str, list[tuple]]` cost ~718 MB; they are now three parallel numpy
arrays in term order at 72 MB. The build is single-pass into typed buffers then
one argsort, so peak build memory stays near the finished index size.

**Latency: 617 ms → 31 ms p50.** The first full-scale measurement showed a 6.4 s
tail. Profiling put the cost in three Python loops that scale with corpus size:
ranking every candidate (56 ms per leg), section priors per candidate (~690 ms),
and metadata filtering through a generator.

*Fixes, in order of impact:* RRF only weights the head of each list, so each leg
is partitioned to its top ~100 with `argpartition` and all fusion, priors and
diversity work happens on that bounded pool; metadata filters became vectorised
compares against integer-coded arrays precomputed per snapshot; score
accumulation moved from `np.add.at` to `np.bincount`.

**Net result: a 21× larger corpus retrieves marginally *faster* than the small
one** (31 ms vs 37 ms p50), because the work is bounded by pool size rather than
corpus size.

Evaluation needed the same treatment: resolving relevance predicates by scanning
every chunk cost minutes across 193 cases × 5 configurations. Label sets are now
narrowed by the vectorised filters before substring matching, and cached.

---

## 11. Embedding experiment: a negative result

The documented bottleneck was the hashed embedder: paraphrase queries collapsed
to ~0.1 nDCG, and dense/lexical had converged at scale. The obvious fix was a
real embedding model. A pluggable layer was built (`hashed`, `ollama`, `openai`,
`google`), and `qwen3-embedding:0.6b` was evaluated locally.

### Method

A 4,926-chunk sub-corpus (MSFT, ORCL, AAPL + the four synthetic issuers), fully
re-embedded, ablation and paraphrase probe run before and after. 53 labelled
cases.

### Result

| Configuration | nDCG@8 hashed | nDCG@8 qwen3 | Δ |
|---|---|---|---|
| lexical only | 0.6766 | 0.6766 | — |
| dense only | 0.5988 | 0.4995 | **−0.099** |
| hybrid 50/50 | 0.6393 | 0.5859 | −0.053 |
| shipped default | 0.6683 | 0.6560 | −0.012 |

Paraphrase probe (dense only): 0.080 → 0.072. **The real model was worse on
every measure.**

### Ruling out the obvious explanations

1. **Integration bug?** No. Self-retrieval is exact: embedding a chunk's own text
   and searching returns that chunk with similarity 1.0000 and correct argmax.
2. **Query/document asymmetry?** Partly. Qwen3-Embedding is instruction-tuned and
   expects a task instruction on the query side only. This was implemented
   (`Embedder.embed_query`, with per-model instruction prefixes for Qwen3/E5/BGE
   families) and improved dense-only from 0.4856 to 0.4995 — real, but nowhere
   near closing the gap.
3. **"The hashed embedder is just a second lexical ranker, so keyword labels
   favour it."** This was the leading hypothesis. **It was tested and is false.**
   Measured top-20 overlap with the BM25 ranking across 10 queries: hashed
   **0.360**, qwen3 **0.390**. The real model is if anything *more*
   BM25-aligned. The hypothesis is dead.

### Conclusion

**The default was not changed.** Shipping `ollama` as default would not be
supported by evidence.

The honest statement of what is known: on this evaluation, with these labels, a
real embedding model does not improve retrieval, and **there is no validated
explanation for why**. The plumbing is verified correct and the asymmetry fix is
implemented.

What this experiment did establish is a **methodological limit**: the paraphrase
probe removes label bias from the *query* side but not from the *labels
themselves*, which still require specific literal terms. The current harness
therefore cannot adjudicate lexical versus semantic retrieval in either
direction. Human-judged relevance on a subset moves from "roadmap item" to
**blocking prerequisite for any further retrieval work**.

The embedding layer ships anyway: it is tested, the migration path
(`creditlens reembed`) is resumable and batched, and the question becomes
answerable the moment better labels exist. Throughput measured at ~10.6
chunks/s locally, ~85 minutes for the full corpus — a cost deliberately not paid,
because the evidence did not justify it.

> This is the most useful result in the project. It is a worked example of a
> plausible improvement being rejected on measurement, an explanation being
> falsified by a follow-up experiment, and the evaluation harness itself being
> identified as the limiting factor.

---

## 12. Observability

- **Structured JSON logs** carrying ambient run and span ids, so one analysis is
  reconstructable from stdout alone.
- **Tracing** — an in-process tracer, OTel-shaped on purpose
  (`Span.to_otlp_like()`); the full span tree ships with the response on request.
- **Metrics** — a dependency-free registry with Prometheus exposition: analyses
  by outcome, tool calls by tool and outcome, tokens by direction and provider,
  spend, HTTP latency histograms.
- **Cost accounting** per run, persisted with every analysis.
- **Settings fingerprint** — a hash of every answer-affecting knob (including
  provider and embedding model), recorded on each analysis and eval run, so
  results can be reproduced or invalidated.

---

## 13. Testing strategy

**307 tests, 89% line coverage, ~12 s, entirely hermetic** — no network, no key,
no shared state. Every test runs against a temporary database seeded with the
synthetic corpus.

| Suite | Covers |
|---|---|
| `test_finance` | ratio math, guards, TTM, derived concepts, attribution, scorecard |
| `test_ingest` | every XBRL and HTML trap in §3.2, as regression tests |
| `test_retrieval` | BM25, fusion, filters, issuer quota, scaling invariants, embedding providers |
| `test_agent` | tools, ledger, the adversarial verifier suite, budgets, degradation |
| `test_providers` | three wire formats, end-to-end through the real loop with fakes |
| `test_api` | HTTP contract |
| `test_eval` | metric correctness against hand-computed values |
| `test_edgar_client` | rate limiting, caching, retry via a stub transport |
| `test_provenance` | entity resolution, lineage invariants, answer intents, API |
| `test_observability` | tracing, metrics, logging |

Two deliberate choices:

- **Tests are derived from real failures**, not invented scenarios. Every case in
  `test_ingest` is something that actually broke against live SEC data.
- **Provider adapters are tested end-to-end through the real orchestrator** with
  fake clients, because translation + parsing + multi-turn together is what
  breaks when a provider is swapped.

CI gates on floors — numeric accuracy ≥ 0.95, citation validity = 1.00, nDCG@8 ≥
0.50, unsupported rate ≤ 0.05 — running `golden,fixtures` against the synthetic
corpus with no network.

---

## 14. Decisions and rejected alternatives

| Decision | Alternative rejected | Reasoning |
|---|---|---|
| SQLite + numpy scan | Postgres + pgvector | 76k chunks = 179 MB of vectors; an in-process scan answers in 31 ms, faster than a network round trip. `DATABASE_URL` switches to Postgres today; the vector-store boundary is where pgvector slots in past ~10⁶ chunks. |
| Vanilla JS frontend | Next.js | One HTML file, one stylesheet, one 530-line script, served by FastAPI. No build step, no `node_modules`, no second runtime in the container. |
| Manual tool loop | Vendor tool runner | Needs per-turn control no runner exposes: ledger registration between execution and the next request, forced terminal turn, mid-loop engine substitution. A vendor runner would also have welded the agent to that vendor. |
| One agent | Planner/critic multi-agent | For a bounded task, adds latency and failure modes without improving answers, and makes traces much harder to reason about. |
| Forced `submit_analysis` | Parse prose | Makes the output schema a contract rather than a hope. |
| Anchored scorecard | Fitted model | No labelled default data in the repo; a fitted model would be false precision. |
| Fictional demo issuers | Real tickers with synthetic numbers | Inventing financials attributed to real companies produces documents indistinguishable from fabricated filings. |
| Programmatic eval labels | Hand-labelled | Well-defined recall, consistent as the corpus grows, free to re-derive. **Limits identified in §11.** |
| Keep `hashed` default | Ship `ollama` default | §11: not supported by evidence. |
| Report Ford's debt as unavailable | Scrape it from statement tables | Ford splits debt across Automotive and Ford Credit tables; a mis-summed leverage ratio is far worse than an explicit "not available". |

---

## 15. Known limitations

1. **The model path is unmeasured.** No live provider run. Every number here is
   from the deterministic path, and the adapters are verified against fakes.
2. **Ford's consolidated debt is unavailable.** Tagged only at segment level via
   custom extensions, so `companyfacts` has no total. Reported precisely
   (`data_coverage` names the missing concepts and the 12 blocked ratios).
3. **Apple does not tag interest expense separately**, so coverage ratios are
   unavailable for it.
4. **Four issuers of 46 lose Item 1A or Item 7 to layout** — Honeywell, Exxon,
   Prologis, Chevron. Text is retrievable; only the item label is missing.
5. **Retrieval quality is not adjudicable** with the current labels (§11).
6. **The dense leg contributes little**, and replacing it is blocked on (5).
7. **Comparisons align periods by label**, not fiscal calendar.
8. **Scorecard anchors are judgement**, not a fitted model.
9. **The universe is US large-cap.** No non-US issuers, no true high-yield names;
   IFRS is accepted by the normalizer but untested in practice.

---

## 16. Roadmap

In priority order, reflecting what §11 established:

1. **Human-judged relevance labels on a subset.** Now the blocking prerequisite:
   without them, no retrieval change can be justified or rejected on evidence.
2. **Run the agent against a live provider** and publish the offline-vs-model
   delta. A small local model is the ideal first adversary for the verification
   layer, which has never faced a generator that can fabricate.
3. **Re-evaluate real embeddings** once (1) exists.
4. **XBRL instance-document parsing with dimension resolution**, to recover
   custom-extension facts such as Ford's debt.
5. **Cross-encoder reranker** over the fused candidate set.
6. **LLM-judge faithfulness metric** to complement the mechanical verifier.
7. **Incremental ingestion** driven by EDGAR's RSS feed.

---

*CreditLens is an analytical tool, not investment advice. The scorecard is an
internal heuristic model and is not a credit rating. The four demo issuers
(NVCR, ARMT, KSTR, HRBG) are fictional and their financials are synthetic.*
