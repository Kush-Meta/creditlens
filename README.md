# CreditLens

**Evidence-grounded credit analysis of public companies, from SEC filings and XBRL data.**

CreditLens answers questions a junior credit analyst would be asked — *"has Oracle's
leverage improved?"*, *"what caused Microsoft's operating margin to move?"*,
*"compare Oracle and Microsoft's liquidity"* — and returns a credit direction, the
metrics behind it, the drivers, the risks, and a citation for every qualitative claim.

The design principle is one sentence:

> **The language model reasons and explains. Ordinary code computes.**

No ratio, growth rate, difference, or sum in a CreditLens answer is produced by a
model. Every figure comes from a deterministic calculation engine, is registered in
an evidence ledger before it re-enters the conversation, and is **checked back
against that ledger after the answer is written**. Numbers the model could not have
got from a tool are surfaced as unsupported rather than shipped.

---

## Contents

- [Quickstart](#quickstart)
- [What it does](#what-it-does)
- [The issuer universe](#the-issuer-universe)
- [Architecture](#architecture)
- [The data pipeline, and what real filings do to it](#the-data-pipeline-and-what-real-filings-do-to-it)
- [Scaling: what 21× more corpus broke](#scaling-what-21-more-corpus-broke)
- [Deterministic financial engine](#deterministic-financial-engine)
- [Retrieval](#retrieval)
- [Agent](#agent)
- [LLM providers](#llm-providers)
- [Verification](#verification)
- [Evaluation and measured results](#evaluation-and-measured-results)
- [Observability](#observability)
- [API](#api)
- [Testing](#testing)
- [Deployment](#deployment)
- [Technology choices](#technology-choices)
- [Known limitations](#known-limitations)

---

## Quickstart

```bash
pip install -e ".[dev]"
```

Load the offline demo corpus — four **fictional** issuers with synthetic financials,
no network and no API key required:

```bash
python -m creditlens.cli seed
```

Ingest the curated 42-issuer universe from SEC EDGAR (~5 minutes, 4 parallel
fetchers under a shared rate limiter):

```bash
make universe
```

Or a single company:

```bash
python -m creditlens.cli ingest --ticker MSFT --min-year 2021
```

Ask a question:

```bash
python -m creditlens.cli ask "Compare the liquidity of Oracle and Microsoft"
```

Run the API and web UI at <http://localhost:8000>:

```bash
python -m creditlens.cli serve
```

Or the whole demo in one step — seeds the fixtures and ingests the full universe:

```bash
make seed && make universe && make serve
```

**Any provider, or none.** CreditLens is not tied to one vendor — see
[LLM providers](#llm-providers). With no credentials at all it falls back to a
deterministic planner/synthesizer, labels the answer `engine:
offline-deterministic`, and says so in the response. That is what makes the test
suite, CI and the evaluation harness runnable with no key and no network.

```bash
python -m creditlens.cli providers          # what is usable from this machine
python -m creditlens.cli ask "..." --provider openai --model gpt-5
```

---

## What it does

A question goes in; a structured, checked analysis comes out:

| Field | Source |
|---|---|
| `credit_direction` | `improving` / `stable` / `deteriorating` / `mixed` / `not_applicable` |
| `key_metrics` | computed by the ratio engine, never by the model |
| `positive_factors`, `risk_factors` | model synthesis, each carrying a citation |
| `reasoning` | model narrative over the tool results |
| `citations` | filing passages with issuer, form, item, date and a link to sec.gov |
| `verification` | per-figure grounding status, citation validity, confidence |
| `usage`, `estimated_cost_usd`, `latency_ms`, `trace` | full run accounting |

The web UI exposes the same thing plus a metric explorer (all 29 ratios, the credit
scorecard, data-coverage report), filing search with per-hit score decomposition,
and an evaluation runner.

---

## The issuer universe

42 real issuers, selected for **credit dispersion rather than market
capitalisation**. A corpus of mega-cap technology names is broad but
analytically flat: leverage and coverage cluster in a narrow band, so neither a
ratio engine nor a scorecard can be seen to differentiate.

| Sampling profile | n | What it contributes |
|---|---|---|
| `net_cash` | 7 | cash exceeds debt — MSFT, AAPL, GOOGL, NVDA, ADBE, JNJ, TSLA |
| `investment_grade` | 12 | moderate leverage, comfortable coverage |
| `leveraged` | 11 | debt-funded M&A or capital return — ORCL, CHTR, CVS, OXY, KHC |
| `structurally_levered` | 6 | utilities and REITs, where high leverage is normal |
| `cyclical_stressed` | 6 | airlines, cruise, weak retail — BA, CCL, KSS, M |

Across 11 sectors, 12 filings each: **516 filings, 76,758 chunks, 34,066
financial facts**. `profile` is a *sampling label recording why an issuer is in
the universe* — never a rating, never an assessment. The credit view is computed
from each issuer's own filings, and comparing the two is a sanity check on the
scorecard, not a target to fit it to.

Ingestion fans out network fetches across a thread pool while persistence stays
on one thread: EDGAR is dominated by download latency, SQLite tolerates exactly
one writer, and the rate limiter is shared so parallelism never breaches the
SEC's request budget. Runs are resumable — an issuer already holding its full
filing count is skipped.

---

## Architecture

```
                       ┌───────────────────────────────────────────┐
   SEC EDGAR ─────────▶│ INGESTION                                 │
   companyfacts XBRL   │  edgar_client   rate limit · cache · retry│
   10-K / 10-Q HTML    │  xbrl_normalizer fiscal calendar · dedup  │
                       │                  YTD→quarter · Q4 rebuild │
                       │  filing_parser   item sections            │
                       │  chunker         paragraph-aware + overlap│
                       └───────────────┬───────────────────────────┘
                                       ▼
                       ┌───────────────────────────────────────────┐
                       │ STORE   facts · filings · chunks · vectors│
                       │         SQLAlchemy → SQLite or Postgres   │
                       └───────┬───────────────────────┬───────────┘
                               ▼                       ▼
          ┌────────────────────────────┐  ┌────────────────────────────┐
          │ FINANCE (deterministic)    │  │ RETRIEVAL                  │
          │  taxonomy   XBRL→concepts  │  │  BM25 + dense embeddings   │
          │  statements periods · TTM  │  │  RRF fusion · section prior│
          │  ratios     29, with guards│  │  metadata filters · quota  │
          │  trends     exact attribution│ │                            │
          │  scorecard  weighted + Z'' │  │                            │
          └─────────────┬──────────────┘  └──────────────┬─────────────┘
                        └────────────┬──────────────────┘
                                     ▼
                       ┌───────────────────────────────────────────┐
                       │ AGENT   10 tools · one controlled loop    │
                       │  every tool result → EVIDENCE LEDGER      │
                       └───────────────┬───────────────────────────┘
                                       ▼
                       ┌───────────────────────────────────────────┐
                       │ VERIFIER  extract every figure from the   │
                       │  answer → match against the ledger →      │
                       │  verified / quoted / contradicted /       │
                       │  unsupported · citation validity          │
                       └───────────────┬───────────────────────────┘
                                       ▼
                          FastAPI  ·  Web UI  ·  CLI  ·  Eval
```

Everything the agent can do is also a plain HTTP endpoint, so the deterministic
layers can be tested, benchmarked and used without paying for an LLM call.

---

## The data pipeline, and what real filings do to it

Most of the engineering here is in the gap between "XBRL is structured data" and
what filers actually publish. Each of the following broke the pipeline against real
SEC data and is now a fixed behaviour with a regression test.

**1. `fy` describes the filing, not the fact.** Every 10-K restates two prior years
as comparatives, and all three carry the *filing's* fiscal year. Trusting it files
FY2023 revenue under FY2025 — and because the dedup key includes the fiscal year, it
silently mixes periods inside one "year". CreditLens ignores `fy` for period
assignment and derives the label from the period end date, using a **fiscal calendar
inferred per company** (modal year-end month, plus a year offset calibrated from
filings whose `fy` *is* trustworthy). This handles Oracle (FY ends May 2026 =
FY2026), Apple (late September), Microsoft (June) and retail calendars (FY ending
January 2025 = fiscal 2024) with the same code.

**2. A quarter ending September is not FY2025 for a June-year-end filer.** Period
labels roll into the fiscal year that *closes* on or after them. Getting this wrong
produced non-contiguous quarter series for every non-calendar filer.

**3. 52/53-week calendars drift past month boundaries.** Apple's year end can land on
1 October. Quarter assignment measures distance to the year end in days and snaps to
the nearest quarter, with a 10-day tolerance on the rollover.

**4. 10-Q cash-flow statements are cumulative.** The Q3 filing reports nine months,
not three. Without differencing (`Q2 = YTD6 − Q1`, `Q3 = YTD9 − YTD6`), quarterly
operating cash flow, capex and D&A simply do not exist — which removes EBITDA, free
cash flow and every ratio built on them from the TTM view. Derived quarters are
marked `derived-from-ytd`.

**5. Nobody files a 10-Q for Q4.** Quarterly flow series have a structural hole,
filled as `Q4 = FY − (Q1+Q2+Q3)` for flow concepts only, marked `derived-q4`.

**6. Year-end balance sheets belong to both Q4 and FY.** Otherwise annual ratio views
have income-statement data and no balance sheet.

**7. Filers tag the same quantity differently.** `Revenues` vs
`RevenueFromContractWithCustomerExcludingAssessedTax`; Microsoft and Oracle never
emit a combined D&A tag and report `Depreciation` + `AmortizationOfIntangibleAssets`
separately. A canonical concept taxonomy with **priority-ordered aliases** resolves
this, and the losing tag is retained in `raw_concept` for audit.

**8. `LongTermDebt` already includes current maturities at some filers.** Adding
short-term debt on top double-counts. Total debt has three paths in trust order: a
filer-reported combined total, components summed, or an inclusive long-term tag used
alone — each recording which path it took.

**9. Filing HTML fights back.** Every item heading appears in the table of contents
before it appears as a heading, and Microsoft prints "Item 7" atop *every page* of
the MD&A — 15 matches inside one section. Section detection drops TOC lines by gap,
takes the first surviving occurrence per item, ends each section at the *next
selected section*, and picks the **longest canonically-ordered subsequence** of items
so one stray "Item 16" cross-reference near the top cannot truncate the document.
The audited statements are then split out as Item 8 wherever they landed.

**10. Not every filer puts the contents at the front, or repeats item numbers
at all.** GE and Honeywell publish their item cross-index at the *back* of the
10-K, so anchoring section detection after the contents block finds nothing.
Honeywell renders every heading with its page number in the same table row
("30 | ITEM 1A. | Risk Factors"), which no item-number pattern matched. And
Chevron, Exxon and GE introduce MD&A by title alone, never repeating "Item 7"
in the body. Detection now removes the contents *block* wherever it sits,
tolerates page-number prefixes, and falls back to title matching only when the
item-number pass comes up empty — applied as a peer instead of a fallback, title
matching fires on cross-references and makes good parses worse, which is exactly
what happened when it was tried first.

Result on real filings — 10-K item coverage after these fixes:

| Issuer | Sections recovered | Item 7 (MD&A) | Item 8 (statements) |
|---|---|---|---|
| MSFT | 1, 1A, 1C, 2, 5, 7, 7A, 8, 9A, 9B, 10, 15, 16 | 51 KB | 105 KB |
| ORCL | 1, 1A, 1C, 2, 5, 7, 7A, 9A, 10, 15, 8 | 65 KB | 159 KB |
| AAPL | 1, 1A, 1C, 3, 5, 7, 7A, 8, 9A, 9B, 15 | 18 KB | 63 KB |
| F | 1, 1A, 1C, 2, 3, 4A, 5, 7, 7A, 9A, 10, 11, 15, 16, 8 | 161 KB | 227 KB |

Across the full 46-issuer corpus, **42 (91%) yield both Item 1A and Item 7**.
Four do not: Honeywell, Exxon, Prologis and Chevron lose one or both to layouts
the splitter still mis-reads. Their text is ingested and fully retrievable — only
the item label is missing, so item-filtered search degrades for those issuers
while everything else works.

---

## Scaling: what 21× more corpus broke

The universe expansion took the corpus from 3,609 to 76,758 chunks. Three things
that were free at the smaller size were not, and each was found by measurement
rather than by inspection.

**Memory: 1.99 GB → 484 MB.** Profiling projected 2 GB at target scale, dominated
by two avoidable costs. Tokenised text retained on every chunk record accounted
for ~844 MB; it is now built during index construction and discarded. BM25
postings as `dict[str, list[tuple]]` accounted for ~718 MB; they are now three
parallel numpy arrays in term order — doc ids, term frequencies, and per-term
offsets — at 72 MB, an 8.4× reduction. The build is single-pass into typed
buffers then one argsort, so peak build memory stays near the size of the
finished index rather than several times it.

**Latency: 617 ms → 31 ms p50.** The first measurement at full scale showed p50
617 ms and a 6.4 s tail. Profiling put the cost in three Python loops that scale
with corpus size: ranking every candidate (56 ms per leg), evaluating section
priors per candidate (~690 ms at 87k), and metadata filtering through a
generator (3.3 ms). Fixes, in order of impact — reciprocal rank fusion only ever
gives meaningful weight to the head of each ranked list, so each leg is now
partitioned to its top ~100 with `argpartition` and all fusion, priors and
diversity work happens on that bounded pool; metadata filters became vectorised
compares against integer-coded arrays precomputed per snapshot; and BM25 score
accumulation moved from `np.add.at` to `np.bincount`.

**The net result is that a 21× larger corpus retrieves marginally faster than
the small one** (31 ms vs 37 ms p50), because the work is now bounded by the
pool size rather than by the corpus.

| | 4 issuers, 3,609 chunks | 42 issuers, 76,758 chunks |
|---|---|---|
| Retrieval p50 / p95 | 37 / 64 ms | **31 / 44 ms** |
| In-memory index | 19 MB | **484 MB** |
| BM25 index | 3.1 MB | 72 MB |
| Snapshot build | 0.1 s | 3.5 s |
| Full eval suite | 26 cases, 4 s | 193 cases, 95 s |

Evaluation itself needed the same treatment: resolving a relevance predicate by
scanning every chunk's text cost minutes across 193 cases and 5 ablation
configurations. Label sets are now narrowed by the vectorised metadata filters
before any substring matching and cached across configurations.

---

## Deterministic financial engine

**29 credit ratios**, each a declared object rather than an inline expression, so
every result carries its formula, the exact inputs it consumed, and the provenance of
each input. Three properties fall out of that:

- **Guardrails.** A ratio with a meaningless denominator returns `None` *plus a
  reason* — `"ebitda is -412,000,000; the ratio is not economically meaningful with a
  non-positive denominator"` — rather than a number that looks authoritative. Credit
  analysis is full of these traps.
- **Period awareness.** Quarterly flows are annualised where required, with a
  seasonality warning attached; TTM sums flow items over four quarters and takes
  balance-sheet items from the most recent one. Stock/flow confusion is impossible by
  construction (`taxonomy.is_flow` gates every aggregation).
- **Introspection.** The agent can list available ratios and their inputs instead of
  inventing formulas.

**Change attribution is arithmetic, not narrative.** For any ratio N/D the change
decomposes exactly:

```
N₁/D₁ − N₀/D₀  =  (N₁−N₀)/D₁  −  N₀(D₁−D₀)/(D₀D₁)
                   ╰ numerator ╯   ╰──── denominator ────╯
```

so *"margin fell 180 bps: −240 bps from cost growth, +60 bps from revenue growth"* is
computed and the components provably sum to the total. Margins get a second,
line-by-line bridge across expense lines as a share of revenue, with an explicit
residual for items the filer did not tag separately.

**Credit scorecard.** Ten weighted factors with piecewise-linear anchor curves →
0–100 composite → implied rating band, plus an Altman Z''-score. Weights renormalise
over computable factors and the covered weight share is reported, so a score built
from 40% of the factors never looks as confident as one built from all of them. It is
labelled an internal heuristic model — not a rating, not investment advice — in every
response that contains it.

---

## Retrieval

Hybrid BM25 + dense, fused with Reciprocal Rank Fusion, over item-tagged filing
chunks.

1. **Metadata pre-filter** — ticker, form, fiscal year, item. Analyst questions are
   scoped ("the latest 10-K", "Ford"), and filtering before scoring is faster and far
   more precise than hoping the ranker infers scope.
2. **Query expansion** — a curated credit-vocabulary map (`leverage` → `borrowings`,
   `indebtedness`; `liquidity` → `revolving credit facility`, `commercial paper`).
   Additive, and applied only to the lexical leg so a bad expansion cannot hijack the
   dense ranking.
3. **RRF fusion** — fuses by *rank*, not score, so BM25's unbounded scores and
   cosine's [−1, 1] never need calibrating. That is the classic failure mode of naive
   score-weighted hybrid search.
4. **Section priors** — a risk question prefers Item 1A; a margin question prefers
   Item 7. Small multiplicative priors, never large enough to override a strong direct
   match.
5. **Per-issuer quota** — a comparison question is useless if all eight passages come
   from one issuer, which is exactly what a global ranking produces when one company
   phrases the topic more strongly. On the labelled comparison cases this is the
   difference between evidence for both sides and evidence for neither
   (`compare-msft-orcl-liquidity`: nDCG@8 0.000 → 0.174).

Retrieval latency on the 3,609-chunk corpus: **p50 37 ms, p95 64 ms** (in-process
numpy scan + inverted index, no network hop).

---

## Agent

**One agent, one loop, ten tools, hard budgets.** No planner/critic hierarchy and no
dynamic agent spawning: for a bounded analytical task those add latency and failure
modes without improving answers, and they make the trace much harder to reason about.

```
list_companies · get_financials · compute_ratios · compare_periods · metric_trend
credit_scorecard · compare_companies · search_filings · data_coverage
                        ↓
                  submit_analysis   (structured final answer, called exactly once)
```

Guarantees the loop enforces regardless of what the model does:

- every tool result is deterministic code output, registered in the evidence ledger
  before it re-enters the conversation;
- the loop always terminates — iteration cap, tool-call cap, and a forced final turn
  via `tool_choice`;
- an answer is always produced: a model failure degrades to the deterministic engine
  and says so in `degraded_reason`, rather than returning a 500;
- a failed tool call returns a `tool_result` with `is_error` **and the valid
  alternatives**, so the model can recover instead of the turn dying.

The final answer is a forced `submit_analysis` tool call rather than parsed prose,
which makes the output schema a contract instead of a hope.

**Prompt caching** is structural: `tools` → `system` form a byte-stable prefix, and
all volatile content (corpus roster, the question, tool results) goes after it as
user messages. `usage.cache_read_input_tokens` is reported per run so a silent
invalidation is visible.

Model configuration is per provider — `claude-opus-5` with adaptive thinking and
`effort: high` on Anthropic, provider defaults elsewhere. Cost is estimated per
run from published rates with cache reads billed at 0.1×, and reported as
unpriced when no rate is known.

---

## LLM providers

The agent is vendor-neutral. Fourteen providers ship, and adding one is a class
plus a table entry — nothing in the tools, the verifier, the evaluation harness
or the API knows which is in use.

| | Providers |
|---|---|
| Anthropic | `anthropic`, `bedrock`, `vertex` |
| OpenAI protocol | `openai`, `azure`, `groq`, `together`, `openrouter`, `deepseek`, `mistral` |
| Google | `google` (Gemini) |
| Self-hosted | `ollama`, `vllm` — and anything else speaking the OpenAI protocol |
| None | `offline` — the deterministic engine |

### What made this non-trivial

The seam was already there (`LLMEngine` is an ABC, and the offline engine
proved it). What was coupled was the **conversation format**: the loop built
Anthropic-shaped message dicts directly, and the three major tool protocols
disagree in ways that cannot be papered over at the edges.

| | Assistant's tool call | Result goes back as |
|---|---|---|
| Anthropic | `tool_use` block inside message content | `tool_result` blocks in one **user** message |
| OpenAI | `tool_calls` on the assistant message | one message per result, `role="tool"` + `tool_call_id` |
| Google | `functionCall` part | `functionResponse` parts under `role="user"`, correlated **by function name** — no call id exists |

So the loop now speaks in neutral `Turn` objects — a user message, an assistant
turn, or a batch of tool results — and each engine translates the whole
transcript into its own wire format on every call. Engines are stateless with
respect to the conversation, which makes retries, replay, and *substituting the
engine mid-run* all trivially correct. That last one matters: it is how a
provider outage degrades to the deterministic engine in the middle of an
analysis rather than failing the request.

An `AssistantTurn` also carries the provider's own representation of that turn,
replayed verbatim where the provider requires it — Anthropic thinking blocks
must be echoed back unchanged, and reconstructing them from text would silently
drop them.

Smaller details that only show up against real providers, each with a test:

- Google's schema dialect rejects `additionalProperties`, `minItems` and
  friends, so tool schemas are sanitised rather than passed through.
- OpenAI reports cached tokens *inside* `prompt_tokens`; billing both would
  charge the same tokens twice at two different rates.
- Reasoning models take `max_completion_tokens`, not `max_tokens`.
- Weaker local models emit malformed tool-argument JSON. That is surfaced as a
  recoverable tool error, not a crash — the loop already knows how to hand an
  error back to the model with the valid alternatives.
- Keyless local servers cannot be detected by "is a key set". They are probed
  by asking for the model list and requiring a JSON answer — a bare TCP connect
  reports any unrelated web app on port 8000 as a working model server, which is
  exactly what happened on the machine this was built on.

### Cost accounting without inventing numbers

Anthropic rates ship built in. For every other model, CreditLens reports
`cost_basis: "no rate configured"` rather than `$0.00`. In a system whose entire
premise is refusing to state numbers it cannot support, silently inventing a
price would undercut the point. Configure rates explicitly:

```bash
CREDITLENS_MODEL_PRICES="gpt-5=1.25/10,llama-3.3-70b=0.6/0.6"
```

Locally hosted providers are priced at zero, which is true rather than assumed.

### Selection

`auto` (the default) tries providers in order and takes the first whose
credentials actually resolve, falling back to the deterministic engine. A named
provider is pinned. Construction failure never propagates — an unreachable
provider degrades with the reason recorded on the analysis, so the service
answers a narrower question instead of returning an error.

---

## Verification

Prompting a model to "only use numbers from tools" is a request, not a guarantee.
CreditLens turns it into a measurement. Every figure in the answer is extracted with
its unit and classified:

| Status | Meaning |
|---|---|
| `verified` | matches a value in the evidence ledger within tolerance |
| `cited` | not computed, but quoted verbatim from a retrieved passage |
| `contradicted` | near a computed value but outside tolerance — a transcription or rounding error, and the most serious class because it looks plausible |
| `unsupported` | traceable to nothing |

Three details make this work rather than just look like it works:

- **Display precision sets a floor on tolerance.** `"18.6%"` asserts 18.6 ± 0.05, not
  18.600000. Without this, a value the engine produced as 18.63 and the narrative
  faithfully rounded to 18.6 is scored as a contradiction — the single largest source
  of false positives. Fixing it moved the deterministic engine's numeric accuracy
  from 0.62 to 1.00 with no change to the generator.
- **Matching is metric-aware.** Candidates are restricted to evidence whose *own
  sentence* names them, using label tokens plus a phrase vocabulary
  (`"gross leverage"` → `debt_to_ebitda`). Without it, *"the current ratio was 1.29x"*
  happily matches a debt/equity of 1.29 and is scored `verified` — a false pass that
  would make the whole layer meaningless.
- **Quotations outrank fuzzy proximity.** A figure quoted from a cited filing is
  grounded by the citation, even when it happens to sit within 25% of some computed
  value. Grounding requires the numeric token *and* surrounding wording to survive
  verbatim in the passage, checked on both sides independently.

Confidence is derived from these rates plus evidence breadth, and is **capped by, not
taken from,** the model's own stated confidence — a model may lower its confidence but
never raise it above what the evidence supports.

Adversarial suite (in `tests/test_agent.py::TestVerifier`), 10/10 correct:

| Claim | Expected | Result |
|---|---|---|
| `Gross leverage stands at 13.00x` (true) | verified | ✅ |
| `Net leverage is 8.72x` (true) | verified | ✅ |
| `Net leverage is 4.10x` (invented) | unsupported | ✅ |
| `Interest coverage was 9.8x` (invented) | unsupported | ✅ |
| `Revenue reached $47.3 billion` (invented) | unsupported | ✅ |
| `The current ratio was 1.29x` (real number, wrong metric) | not verified | ✅ |
| `Debt to equity is 1.29x` (true) | verified | ✅ |
| `…approximately $2.1 billion of senior notes mature…` (quoted) | cited | ✅ |
| `…disclosed $8.4 billion of senior notes maturing…` (fabricated in the same phrasing) | unsupported | ✅ |
| `See [C99]` | invalid citation | ✅ |

---

## Evaluation and measured results

`creditlens/eval/` holds **209 cases across three suites**:

- **`golden`** — 26 hand-written cases covering the questions that need
  judgement: causal attribution, data-gap honesty, an issuer deliberately absent
  from the corpus.
- **`universe`** — 167 generated cases, so coverage tracks the corpus instead of
  lagging it. A hand-written suite does not scale to 42 issuers, and a system
  measured on four of them is not really measured.
- **`fixtures`** — 16 cases generated the same way from the synthetic issuers,
  so CI can gate on every question family without an EDGAR ingest.

Generation is deterministic (seeded per ticker), assigns every issuer two core
question families plus a rotating pair, and adds one within-sector comparison
per sector — paired across *differing* credit profiles, so the comparison
discriminates rather than matching two lookalikes (Ford vs Tesla, Home Depot vs
Kohl's, Carnival vs Delta, AbbVie vs J&J). Three rules keep generated cases from
becoming filler: every case asserts something checkable, generation is
reproducible, and **cases the corpus cannot score are dropped at build time**
rather than silently failing at run time — 12 were removed because their
relevance predicate matched no chunk, all of them on the four issuers whose
sections parse badly.

**Relevance labels are programmatic.** A case declares issuer, filing item and
the terms a genuinely useful passage must contain; the harness resolves that
predicate against the whole corpus to build the ground-truth relevant set.
Recall is therefore well-defined, labels stay consistent as the corpus grows,
and they cost nothing to re-derive after a re-ingest.

Keyword-defined labels *structurally favour a keyword retriever*. They are
excellent for regression-testing a retrieval change and weak as an absolute
quality score — which is why the lexical-versus-dense comparison below is
followed by a paraphrase probe the labels cannot bias.

**Numeric ground truth is recomputed independently** through the finance layer,
not read from the agent's own tool output — checking the agent against its own
tools would only prove it can copy. Values are compared against the period the
answer *claims*, so reporting TTM instead of FY2026 is tracked as a scope
difference rather than a wrong number. Metrics an issuer genuinely cannot
support are excluded from scoring entirely: Ford tags no consolidated debt, and
penalising an answer for not reporting an uncomputable figure would punish
exactly the honest behaviour the system is built for.

### Measured: end-to-end suite

Corpus: 46 issuers (42 real, 4 synthetic), 516 filings, **76,758 chunks, 34,066
financial facts**. Engine: `offline-deterministic` (see the note below).

| Metric | 26 cases, 4 issuers | **193 cases, 46 issuers** |
|---|---|---|
| Numeric accuracy | 0.968 | **1.000** |
| Unsupported claim rate | 0.032 | **0.000** |
| Citation validity | 1.000 | **1.000** |
| Ground-truth metric accuracy | 0.867 | **0.927** |
| Reported-period match | 0.733 | **0.908** |
| Retrieval nDCG@8 | 0.674 | **0.574** |
| Retrieval recall@8 (ceiling-normalised) | 0.633 | **0.520** |
| Credit-direction accuracy | 1.000 | **1.000** |
| Tool-selection F1 | 0.539 | 0.368 |
| Retrieval latency p50 / p95 | 37 / 64 ms | **31 / 44 ms** |

Numeric accuracy reaching 1.000 across 193 cases is not the generator getting
better — it is the verifier getting more precise. Two false-positive classes
were found only because the larger suite made them visible: a floating-point
boundary where `|1.1 − 1.05|` evaluates to `0.05000000000000004` and failed an
exact comparison against a 0.05 tolerance, and sentence-level metric attribution
being too coarse for prose that lists several metrics in one sentence
("Current ratio 1.09x, Cash / Debt 42.3%, FCF margin 22.7%" — each figure was
attributed to whichever metric scored highest across the whole sentence).
Attribution is now scoped to the comma-delimited clause, widening to the
sentence only when the clause names no metric.

Retrieval nDCG falling from 0.674 to 0.574 is the honest cost of a 21× larger
corpus: the same queries now compete against twenty times more plausible
distractors. Tool-selection F1 falling is a limitation of the deterministic
baseline, which follows a fixed plan rather than selecting a metric-specific
tool from the question — precisely what a model is expected to improve.

**Uniformity across the credit spectrum** — the reason for slicing:

| Sampling profile | cases | numeric | nDCG@8 |
|---|---|---|---|
| cyclical_stressed | 24 | 1.000 | 0.590 |
| investment_grade | 40 | 1.000 | 0.590 |
| leveraged | 44 | 1.000 | 0.567 |
| net_cash | 28 | 1.000 | 0.567 |
| structurally_levered | 20 | 1.000 | 0.585 |

Numeric grounding is uniform; retrieval varies by sector (Autos 0.716 and
Utilities 0.707 at the top, Energy 0.401 and Travel 0.476 at the bottom), which
tracks section-parse quality rather than anything about the questions.

### Measured: retrieval ablation

192 labelled cases, k = 8:

| Configuration | recall@k (norm.) | precision@k | MRR | nDCG@8 |
|---|---|---|---|---|
| lexical only (BM25) | 0.530 | 0.525 | 0.804 | **0.581** |
| dense only | 0.520 | 0.515 | 0.795 | 0.574 |
| hybrid RRF 50/50 | 0.518 | 0.514 | 0.800 | 0.571 |
| hybrid RRF + MMR (λ=0.7) | 0.439 | 0.435 | 0.823 | 0.482 |
| **shipped default** (dense 0.2, no MMR) | 0.520 | 0.516 | 0.796 | 0.574 |

Two decisions come from this table rather than from taste:

- **MMR is off by default.** It costs 0.10 nDCG@8 at this scale (0.581 → 0.482),
  and cost 0.24 on the smaller corpus. Relevant passages in filings genuinely
  cluster — four quarters of the same liquidity discussion are all relevant — so
  diversity trades away hits. It stays available per request.
- **Dense weight is 0.2, not 0.5.** On keyword queries lexical-only still ranks
  highest, but the gap has closed sharply with scale: 0.687 vs 0.590 on four
  issuers, 0.581 vs 0.574 on forty-two. On a paraphrase probe — eight queries
  deliberately avoiding the terms that define the labels — the ordering flips:

  | Configuration | nDCG@8 on paraphrased queries |
  |---|---|
  | Lexical only | 0.092 |
  | Dense only | 0.080 |
  | **Hybrid, dense 0.2** | **0.115** |
  | Hybrid, dense 0.5 | 0.113 |

  The probe still tells the real story: **the hashed embedder is the weak link** —
  everything collapses to roughly 0.1 nDCG on paraphrases. Swapping in a hosted
  embedding model is one method implementation, and remains the single
  highest-leverage improvement available.

### CI gate

CI runs the suite offline on every push and fails on regression against floors:
numeric accuracy ≥ 0.95, citation validity = 1.00, retrieval nDCG@8 ≥ 0.50,
direction accuracy ≥ 0.90, unsupported-claim rate ≤ 0.05. CI runs `golden,fixtures`
against the synthetic corpus, which needs no network; the `universe` suite
requires an EDGAR ingest and runs on demand.

---

## Observability

- **Structured JSON logs** carrying the ambient run and span id, so one analysis can
  be reconstructed end-to-end from stdout alone.
- **Tracing** — an in-process tracer whose span model is OTel-shaped on purpose
  (`Span.to_otlp_like()`); swapping in a real exporter is a ~30-line change. The full
  span tree ships with the API response when `include_trace` is set: which tools ran,
  in what order, how long each took, with what arguments.
- **Metrics** — a dependency-free registry with a Prometheus text exposition at
  `/metrics` and a JSON snapshot at `/api/metrics`: analyses by outcome, tool calls by
  tool and outcome, LLM tokens by direction, estimated spend, HTTP latency histograms.
- **Cost accounting** per run: input/output/cache-read/cache-write tokens and an
  estimated dollar cost, persisted with every analysis.
- **Settings fingerprint** — a hash of every knob that can change an answer, recorded
  on each analysis and eval run, so results can be reproduced or invalidated.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | database, corpus counts, LLM reachability, fingerprint |
| `GET` | `/metrics`, `/api/metrics` | Prometheus text / JSON snapshot |
| `GET` | `/api/config` | effective configuration and the full ratio catalog |
| `GET` | `/api/companies` | issuers and periods held |
| `GET` | `/api/companies/{t}/financials` | line items across periods, incl. TTM |
| `GET` | `/api/companies/{t}/ratios` | all 29 ratios with formulas and provenance |
| `GET` | `/api/companies/{t}/trend` | series, OLS slope, R², direction |
| `GET` | `/api/companies/{t}/scorecard` | weighted scorecard + Altman Z'' |
| `GET` | `/api/companies/{t}/coverage` | what is missing and which ratios it blocks |
| `POST` | `/api/compare` | multi-issuer ratio and scorecard comparison |
| `POST` | `/api/search` | hybrid retrieval with score decomposition |
| `POST` | `/api/analyze` | full agent analysis |
| `GET` | `/api/runs`, `/api/runs/{id}` | analysis history with verification metrics |
| `POST` | `/api/ingest`, `/api/ingest/fixtures` | ingest an issuer / load the demo corpus |
| `POST` | `/api/eval/run`, `GET /api/eval/runs` | run and browse evaluations |

OpenAPI docs at `/docs`.

---

## Testing

```bash
make test        # 244 tests, ~9 s
make test-cov    # coverage report
make check       # lint + tests, what CI runs
```

**244 tests, 89% line coverage**, and the whole suite is hermetic — no network, no API
key, no shared state. Every test runs against a temporary database seeded with the
synthetic corpus.

The tests worth reading are the ones derived from real failures rather than invented
ones: `tests/test_ingest.py` encodes each XBRL and HTML trap described above
(comparatives filed under the wrong year, cumulative YTD spans, running page headers,
stray item cross-references, 52/53-week drift), and
`tests/test_agent.py::TestVerifier` is the adversarial suite that keeps the
verification layer honest.

---

## Deployment

```bash
make docker-build && make docker-run
# or
docker compose up
```

Multi-stage build, non-root runtime user, no compiler in the final image, health check
wired to `/health`. `docker compose --profile postgres up` brings up the Postgres
path; the application code is unchanged (`CREDITLENS_DATABASE_URL` is the only
difference).

Configuration is environment-driven with a `CREDITLENS_` prefix — see `.env.example`.

---

## Technology choices

The brief suggested Postgres/pgvector and Next.js. Both were considered and both were
declined, for reasons rather than convenience:

**SQLite over Postgres+pgvector (Postgres still supported).** The whole corpus is
~3,600 chunks; a 512-dimension float32 matrix is ~7 MB. An in-process numpy scan
answers in 37 ms p50 including BM25 — faster than the network round-trip to a vector
database, with no extra service to run, back up or version. The storage layer is
plain SQLAlchemy and `CREDITLENS_DATABASE_URL` switches to Postgres today; the
`VectorStore` boundary is where pgvector or Qdrant would slot in when the corpus
outgrows memory (roughly 10⁶ chunks). Shipping a database cluster for 7 MB of vectors
would be architecture as decoration.

**Vanilla JS over Next.js.** The UI is one HTML file, one stylesheet and one 500-line
script, served by FastAPI itself. There is no build step, no `node_modules`, no second
runtime in the container, and the whole frontend is auditable in one sitting. It is
theme-aware, responsive, and shows everything the API returns — including the
per-figure verification table, which is the point of the product. A framework would
add tooling without adding capability at this size.

**Hashed n-gram embeddings by default.** Deliberate, and measured (see above):
reproducible in CI, offline, no key, and it makes the *architecture* the thing under
test. `Embedder` is an ABC with one method; a hosted model is a drop-in. The README
reports honestly that this backend contributes little semantic generalisation.

**A manual tool loop over any SDK's tool runner.** The loop needs per-turn
control no vendor runner exposes: evidence-ledger registration between tool
execution and the next request, a forced terminal turn on budget exhaustion, and
mid-loop engine substitution when a provider becomes unavailable. It is also
what makes the loop portable — a vendor runner would have welded the agent to
that vendor.

---

## Known limitations

Stated plainly, because a credit tool that hides its gaps is worse than one that has
them.

1. **Ford's consolidated debt is not available.** Ford tags debt only at segment level
   through custom taxonomy extensions, so the SEC `companyfacts` API — which serves
   standard taxonomies — has no total debt or interest expense for it. CreditLens
   reports this precisely (`data_coverage` names the missing concepts and the 12
   blocked ratios) rather than estimating. Scraping the figure out of the filing's
   statement tables was prototyped and rejected: Ford splits debt across Automotive
   and Ford Credit tables, and a mis-summed leverage ratio is far worse than an
   explicit "not available". The correct fix is parsing the filing's own XBRL instance
   document with dimension resolution — a real subsystem, on the roadmap.
2. **Apple does not tag interest expense separately**, so coverage ratios are
   unavailable for it. Same handling.
3. **Ford's MD&A is attributed to Item 5.** Its Item 7 heading does not survive HTML
   flattening in a form the section splitter recognises. The text is in the corpus and
   fully retrievable; only its item label is wrong, which degrades item-filtered
   search for that one issuer.
4. **The dense retriever is weak.** Hashed n-grams are a lexical model in disguise;
   paraphrase queries collapse to ~0.1 nDCG. Highest-leverage fix available.
5. **The model path is unmeasured here** — no credentials were available. Every number
   in this README comes from the deterministic path.
6. **Programmatic relevance labels favour lexical retrieval.** Disclosed above, and
   partially mitigated by the paraphrase probe.
7. **Comparisons align periods by label, not by fiscal calendar.** Issuers with
   different year ends are not perfectly comparable; the tool says so in its output.
8. **The scorecard anchors are judgement, not a fitted model.** With no labelled
   default data in the repo, a fitted model would be false precision. The anchors are
   auditable line by line and the output is labelled a heuristic, never a rating.

9. **The universe is US large-cap.** No non-US issuers, no true high-yield
   names, and IFRS filers are unsupported by the taxonomy in practice even though
   the normalizer accepts the namespace.

### Roadmap, in priority order

1. A hosted embedding model behind `Embedder` — the measured bottleneck, and now
   the clearest one: dense and lexical have converged at scale, so the dense leg
   is contributing little beyond paraphrase robustness.
2. XBRL instance-document parsing with dimension resolution, to recover
   custom-extension facts (Ford's debt).
3. A cross-encoder reranker over the fused candidate set.
4. Human-labelled relevance judgements on a subset, to calibrate the programmatic ones.
5. An LLM-judge faithfulness metric to complement the mechanical verifier.
6. Per-issuer incremental ingestion driven by EDGAR's RSS feed.

---

## Licence

MIT.

*CreditLens is an analytical tool, not investment advice. The scorecard is an internal
heuristic model and is not a credit rating. The four demo issuers (NVCR, ARMT, KSTR,
HRBG) are fictional and their financials are synthetic.*
