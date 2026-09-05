/* CreditLens front end.
 *
 * Deliberately framework-free: the UI is a thin, honest view over the API, and
 * a build step would add tooling without adding capability. Every number shown
 * comes from a server response; nothing is computed here.
 */
'use strict';

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
};

const state = { companies: [], config: null, activeTicker: null };

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
  return body;
}

/* ------------------------------------------------------------------ header */
async function loadHealth() {
  const pills = $('#status-pills');
  try {
    const health = await api('/health');
    const llm = health.llm || {};
    pills.replaceChildren(
      el('span', { class: `pill ${health.database ? 'ok' : 'warn'}` },
        `db ${health.database ? 'ok' : 'down'}`),
      el('span', { class: 'pill' }, `${health.corpus.companies} issuers`),
      el('span', { class: 'pill' }, `${health.corpus.chunks.toLocaleString()} chunks`),
      el('span', { class: 'pill' }, `${health.corpus.financial_facts.toLocaleString()} facts`),
      el('span', { class: `pill ${llm.reachable ? 'ok' : 'warn'}` },
        llm.reachable ? llm.model : 'deterministic engine'),
      el('span', { class: 'pill', title: 'Settings fingerprint' }, health.settings_fingerprint),
    );
  } catch (error) {
    pills.replaceChildren(el('span', { class: 'pill warn' }, `api: ${error.message}`));
  }
}

/* ----------------------------------------------------------------- sidebar */
async function loadCompanies() {
  const list = $('#company-list');
  try {
    const payload = await api('/api/companies');
    state.companies = payload.companies;
    if (!payload.companies.length) {
      list.replaceChildren(el('div', { class: 'muted' },
        'Empty. Ingest a ticker, or load the demo corpus.'));
      return;
    }
    list.replaceChildren(...payload.companies.map((company) =>
      el('button', {
        class: `company${state.activeTicker === company.ticker ? ' active' : ''}`,
        onclick: () => selectCompany(company.ticker),
      },
        el('div', {},
          el('span', { class: 'ticker' }, company.ticker),
          company.is_synthetic ? el('span', { class: 'badge-synth' }, 'demo') : null),
        el('div', { class: 'meta' }, company.name.slice(0, 30)),
        el('div', { class: 'meta' },
          `${company.latest_period || 'n/a'} · ${company.periods_available.length} periods`),
      )));
    const select = $('#metrics-ticker');
    select.replaceChildren(...payload.companies.map((c) =>
      el('option', { value: c.ticker }, `${c.ticker} — ${c.name.slice(0, 26)}`)));
    renderExamples();
  } catch (error) {
    list.replaceChildren(el('div', { class: 'notice error' }, error.message));
  }
}

function selectCompany(ticker) {
  state.activeTicker = ticker;
  $('#metrics-ticker').value = ticker;
  loadCompanies();
  switchPanel('metrics');
  loadMetrics();
}

function renderExamples() {
  const tickers = state.companies.map((c) => c.ticker);
  // iterate the preference list, not the alphabetical corpus order
  const pick = (preferred) => preferred.find((t) => tickers.includes(t)) || tickers[0];
  const a = pick(['ORCL', 'NVCR', 'F']);
  const b = pick(['MSFT', 'ARMT']);
  if (!a) return;
  const questions = [
    `Has ${a}'s credit quality improved or deteriorated over the last four quarters?`,
    `What caused ${b}'s operating margin to change?`,
    `Compare the liquidity of ${a} and ${b}.`,
    `Calculate ${a}'s debt-to-EBITDA ratio and explain it.`,
    `What risks did ${a} management highlight in the latest 10-K?`,
    `Summarize ${b}'s credit profile.`,
  ];
  $('#examples').replaceChildren(...questions.map((question) =>
    el('button', {
      class: 'example',
      onclick: () => { $('#question').value = question; $('#ask-form').requestSubmit(); },
    }, question)));
}

/* ---------------------------------------------------------------- analysis */
async function runAnalysis(event) {
  event.preventDefault();
  const question = $('#question').value.trim();
  if (!question) return;
  const output = $('#analysis-output');
  const button = $('#ask-btn');
  button.disabled = true;
  output.replaceChildren(el('div', { class: 'card' },
    el('span', { class: 'spinner' }), 'Gathering evidence and computing metrics…'));
  try {
    const result = await api('/api/analyze', {
      method: 'POST',
      body: JSON.stringify({ question, include_trace: false }),
    });
    renderAnalysis(result);
  } catch (error) {
    output.replaceChildren(el('div', { class: 'notice error' }, error.message));
  } finally {
    button.disabled = false;
  }
}

function renderAnalysis(result) {
  const output = $('#analysis-output');
  const verification = result.verification || {};
  const nodes = [];

  if (result.contains_synthetic_data) {
    nodes.push(el('div', { class: 'notice' },
      'This answer draws on synthetic demo data from fictional issuers. The figures are not filed results.'));
  }
  if (result.degraded_reason) {
    nodes.push(el('div', { class: 'notice' }, result.degraded_reason));
  }

  nodes.push(el('div', { class: 'card' },
    el('div', { style: 'display:flex;gap:12px;align-items:center;flex-wrap:wrap' },
      el('span', { class: `direction ${result.credit_direction}` },
        result.credit_direction.replace(/_/g, ' ')),
      el('span', { class: 'muted' },
        `confidence ${verification.confidence} · ${result.engine} · ${Math.round(result.latency_ms)} ms · $${(result.estimated_cost_usd || 0).toFixed(4)}`)),
    el('p', { class: 'answer' }, result.answer)));

  if (result.key_metrics?.length) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, 'Key metrics'),
      el('div', { class: 'metrics-grid' }, result.key_metrics.map((metric) =>
        el('div', { class: 'metric-tile' },
          el('div', { class: 'label' }, metric.name),
          el('div', { class: 'value' }, metric.value),
          el('div', { class: 'sub' },
            [metric.ticker, metric.period].filter(Boolean).join(' · ')))))));
  }

  const columns = [
    ['Positive factors', result.positive_factors],
    ['Risk factors', result.risk_factors],
  ].filter(([, items]) => items?.length);
  for (const [title, items] of columns) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, title),
      el('ul', { class: 'factors' }, items.map((item) => el('li', {}, item)))));
  }

  if (result.reasoning) {
    nodes.push(el('div', { class: 'card reasoning' },
      el('h3', {}, 'Reasoning'),
      ...result.reasoning.split(/\n{2,}/).map((para) => el('p', {}, para))));
  }

  if (result.caveats?.length) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, 'Caveats'),
      el('ul', { class: 'factors' }, result.caveats.map((c) => el('li', {}, c)))));
  }

  if (result.citations?.length) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, `Citations (${result.citations.length})`),
      ...result.citations.map((citation) => el('div', { class: 'cite' },
        el('div', { class: 'head' },
          `[${citation.label}] ${citation.ticker} · ${citation.source} · item ${citation.item || 'n/a'} · ${citation.filing_date}`),
        el('div', { class: 'snippet' }, citation.snippet),
        citation.url
          ? el('div', {}, el('a', { href: citation.url, target: '_blank', rel: 'noopener' },
              'open filing on sec.gov'))
          : null))));
  }

  nodes.push(renderVerification(verification));

  if (result.tool_calls?.length) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, `Tool calls (${result.tool_calls.length})`),
      el('ul', { class: 'timeline' }, result.tool_calls.map((call) =>
        el('li', { class: call.is_error ? 'err' : '' },
          el('span', { class: 'tool' }, call.tool),
          el('span', { class: 'args' }, JSON.stringify(call.arguments)),
          el('span', { class: 'dur' }, `${call.duration_ms.toFixed(0)}ms`))))));
  }

  output.replaceChildren(...nodes);
}

function renderVerification(verification) {
  const claims = verification.claims || [];
  const counts = {
    verified: verification.numeric_claims_verified || 0,
    cited: verification.numeric_claims_quoted_from_filings || 0,
    contradicted: verification.numeric_claims_contradicted || 0,
    unsupported: verification.numeric_claims_unsupported || 0,
  };
  return el('div', { class: 'card' },
    el('h3', {}, 'Verification'),
    el('div', { class: 'metrics-grid', style: 'margin-bottom:14px' },
      el('div', { class: 'metric-tile' },
        el('div', { class: 'label' }, 'Numeric accuracy'),
        el('div', { class: 'value' }, `${Math.round((verification.numeric_accuracy || 0) * 100)}%`),
        el('div', { class: 'sub' }, `${counts.verified} computed · ${counts.cited} quoted`)),
      el('div', { class: 'metric-tile' },
        el('div', { class: 'label' }, 'Unsupported rate'),
        el('div', { class: 'value' }, `${Math.round((verification.unsupported_claim_rate || 0) * 100)}%`),
        el('div', { class: 'sub' }, `${counts.unsupported} untraced · ${counts.contradicted} off-tolerance`)),
      el('div', { class: 'metric-tile' },
        el('div', { class: 'label' }, 'Citation validity'),
        el('div', { class: 'value' }, `${Math.round((verification.citation_validity || 0) * 100)}%`),
        el('div', { class: 'sub' }, `${(verification.invalid_citations || []).length} invalid`))),
    (verification.notes || []).length
      ? el('ul', { class: 'factors muted' }, verification.notes.map((n) => el('li', {}, n)))
      : null,
    claims.length
      ? el('details', {},
          el('summary', { class: 'muted', style: 'cursor:pointer' },
            `every figure checked (${claims.length})`),
          el('div', { class: 'scroll-x', style: 'margin-top:10px' },
            el('table', {},
              el('thead', {}, el('tr', {},
                el('th', {}, 'Figure'), el('th', {}, 'Status'),
                el('th', {}, 'Traced to'), el('th', {}, 'Context'))),
              el('tbody', {}, claims.map((claim) => el('tr', {},
                el('td', { class: 'num' }, claim.claim),
                el('td', {}, el('span', { class: `claim-status ${claim.status}` }, claim.status)),
                el('td', { class: 'muted' }, claim.matched_evidence || '—'),
                el('td', { class: 'muted' }, (claim.context || '').slice(0, 110))))))))
      : null);
}

/* ----------------------------------------------------------------- metrics */
async function loadMetrics() {
  const ticker = $('#metrics-ticker').value;
  const period = $('#metrics-period').value;
  if (!ticker) return;
  const output = $('#metrics-output');
  output.replaceChildren(el('div', {}, el('span', { class: 'spinner' }), 'Computing…'));
  try {
    const [ratios, scorecard, coverage] = await Promise.all([
      api(`/api/companies/${ticker}/ratios?period=${encodeURIComponent(period)}`),
      api(`/api/companies/${ticker}/scorecard?period=${encodeURIComponent(period)}`).catch(() => null),
      api(`/api/companies/${ticker}/coverage`).catch(() => null),
    ]);
    renderMetrics(ratios, scorecard, coverage);
  } catch (error) {
    output.replaceChildren(el('div', { class: 'notice error' }, error.message));
  }
}

function renderMetrics(ratios, scorecard, coverage) {
  const nodes = [];
  if (scorecard && scorecard.composite_score_0_100 !== null) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, 'Internal scorecard'),
      el('div', { class: 'metrics-grid', style: 'margin-bottom:14px' },
        el('div', { class: 'metric-tile' },
          el('div', { class: 'label' }, 'Composite'),
          el('div', { class: 'value' }, scorecard.composite_score_0_100),
          el('div', { class: 'sub' }, 'out of 100')),
        el('div', { class: 'metric-tile' },
          el('div', { class: 'label' }, 'Implied band'),
          el('div', { class: 'value' }, scorecard.implied_band),
          el('div', { class: 'sub' }, `${scorecard.confidence} confidence`)),
        el('div', { class: 'metric-tile' },
          el('div', { class: 'label' }, "Altman Z''"),
          el('div', { class: 'value' },
            scorecard.altman_z?.available ? scorecard.altman_z.score : 'n/a'),
          el('div', { class: 'sub' }, scorecard.altman_z?.zone || 'inputs missing'))),
      el('div', {}, scorecard.factors.filter((f) => f.score_0_100 !== null).map((factor) =>
        el('div', { class: 'bar-row', title: factor.rationale },
          el('span', { class: 'name' }, factor.label),
          el('span', { class: 'bar-track' },
            el('span', { class: 'bar-fill', style: `width:${factor.score_0_100}%` })),
          el('span', { class: 'val' }, factor.display)))),
      el('div', { class: 'muted', style: 'margin-top:10px' }, scorecard.disclaimer)));
  }

  const byCategory = {};
  for (const ratio of ratios.ratios) (byCategory[ratio.category] ??= []).push(ratio);
  for (const [category, rows] of Object.entries(byCategory)) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, `${category.replace(/_/g, ' ')} — ${ratios.ticker} ${ratios.period}`),
      el('div', { class: 'scroll-x' }, el('table', {},
        el('thead', {}, el('tr', {},
          el('th', {}, 'Ratio'), el('th', { class: 'num' }, 'Value'),
          el('th', {}, 'Formula'), el('th', {}, 'Note'))),
        el('tbody', {}, rows.map((ratio) => el('tr', {},
          el('td', {}, ratio.label),
          el('td', { class: 'num' }, ratio.display),
          el('td', { class: 'muted' }, ratio.formula),
          el('td', { class: 'muted' },
            ratio.unavailable_reason || (ratio.warnings || []).join('; ') || ''))))))));
  }

  if (coverage?.concepts_missing?.length) {
    nodes.push(el('div', { class: 'card' },
      el('h3', {}, 'Data coverage'),
      el('div', { class: 'notice' },
        `${coverage.ticker} does not report ${coverage.concepts_missing.join(', ')} in its XBRL facts, ` +
        `so ${coverage.ratios_blocked.length} ratio(s) cannot be computed.`),
      el('div', { class: 'muted' }, coverage.guidance)));
  }
  $('#metrics-output').replaceChildren(...nodes);
}

/* ------------------------------------------------------------------ search */
async function runSearch(event) {
  event.preventDefault();
  const query = $('#search-query').value.trim();
  if (!query) return;
  const item = $('#search-item').value;
  const output = $('#search-output');
  output.replaceChildren(el('div', {}, el('span', { class: 'spinner' }), 'Searching…'));
  try {
    const result = await api('/api/search', {
      method: 'POST',
      body: JSON.stringify({
        query, top_k: 10, include_text: true,
        items: item ? [item] : null,
      }),
    });
    if (!result.hits.length) {
      output.replaceChildren(el('div', { class: 'notice' },
        `No matches. ${result.diagnostics.reason || ''}`));
      return;
    }
    output.replaceChildren(
      el('div', { class: 'muted', style: 'margin-bottom:14px' },
        `${result.hits.length} of ${result.diagnostics.candidates} candidates · ` +
        `${result.diagnostics.fusion} · dense weight ${result.diagnostics.dense_weight}` +
        (result.diagnostics.expansion_terms?.length
          ? ` · expanded with: ${result.diagnostics.expansion_terms.join(', ')}` : '')),
      ...result.hits.map((hit) => el('div', { class: 'card' },
        el('div', { class: 'head', style: 'font-family:var(--mono);font-size:11.5px;color:var(--text-dim)' },
          `${hit.rank}. ${hit.citation.ticker} · ${hit.citation.form_type} ${hit.citation.period} · ` +
          `item ${hit.citation.item || 'n/a'} · score ${hit.score.toFixed(4)} ` +
          `(dense #${hit.scores.dense_rank ?? '–'}, lexical #${hit.scores.lexical_rank ?? '–'})`),
        el('div', { style: 'margin-top:8px' }, hit.snippet),
        el('details', {}, el('summary', { class: 'muted' }, 'full passage'),
          el('pre', {}, hit.text)))));
  } catch (error) {
    output.replaceChildren(el('div', { class: 'notice error' }, error.message));
  }
}

/* -------------------------------------------------------------- evaluation */
async function runEval() {
  const output = $('#eval-output');
  const button = $('#eval-btn');
  button.disabled = true;
  output.replaceChildren(el('div', { class: 'card' },
    el('span', { class: 'spinner' }), 'Running the golden suite…'));
  try {
    const result = await api('/api/eval/run', {
      method: 'POST',
      body: JSON.stringify({ suite: 'golden', engine: 'offline' }),
    });
    renderEval(result);
  } catch (error) {
    output.replaceChildren(el('div', { class: 'notice error' }, error.message));
  } finally {
    button.disabled = false;
  }
}

function renderEval(result) {
  const headline = result.metrics.headline || {};
  const format = (value) =>
    value === null || value === undefined ? 'n/a'
      : typeof value === 'number' ? (Math.abs(value) < 10 ? value.toFixed(3) : value.toFixed(1))
      : value;
  $('#eval-output').replaceChildren(
    el('div', { class: 'card' },
      el('h3', {}, `${result.suite} · ${result.n_cases} cases · ${result.engine}`),
      el('div', { class: 'metrics-grid' }, Object.entries(headline).map(([key, value]) =>
        el('div', { class: 'metric-tile' },
          el('div', { class: 'label' }, key.replace(/_/g, ' ')),
          el('div', { class: 'value' }, format(value)))))),
    el('div', { class: 'card' },
      el('h3', {}, 'Per case'),
      el('div', { class: 'scroll-x' }, el('table', {},
        el('thead', {}, el('tr', {},
          el('th', {}, 'Case'), el('th', {}, 'Category'),
          el('th', { class: 'num' }, 'Numeric'), el('th', { class: 'num' }, 'nDCG@k'),
          el('th', { class: 'num' }, 'Tool F1'), el('th', { class: 'num' }, 'ms'),
          el('th', {}, 'Findings'))),
        el('tbody', {}, result.results.map((row) => el('tr', {},
          el('td', {}, row.case_id),
          el('td', { class: 'muted' }, row.category),
          el('td', { class: 'num' }, format(row.metrics.numeric_accuracy)),
          el('td', { class: 'num' }, format(row.metrics.retrieval_ndcg_at_k)),
          el('td', { class: 'num' }, format(row.metrics.tool_selection_f1)),
          el('td', { class: 'num' }, Math.round(row.latency_ms)),
          el('td', { class: 'muted' }, (row.failures || []).join(' · ') || '—')))))))
  );
}

async function loadEvalHistory() {
  const output = $('#eval-output');
  try {
    const payload = await api('/api/eval/runs?limit=10');
    if (!payload.runs.length) {
      output.replaceChildren(el('div', { class: 'notice' }, 'No eval runs recorded yet.'));
      return;
    }
    output.replaceChildren(el('div', { class: 'card' },
      el('h3', {}, 'Recent eval runs'),
      el('div', { class: 'scroll-x' }, el('table', {},
        el('thead', {}, el('tr', {},
          el('th', {}, 'When'), el('th', {}, 'Engine'), el('th', { class: 'num' }, 'Cases'),
          el('th', { class: 'num' }, 'Numeric'), el('th', { class: 'num' }, 'nDCG@k'),
          el('th', { class: 'num' }, 'Cost'), el('th', {}, 'Fingerprint'))),
        el('tbody', {}, payload.runs.map((run) => {
          const headline = run.metrics?.headline || {};
          return el('tr', {},
            el('td', { class: 'muted' }, run.created_at.slice(0, 19).replace('T', ' ')),
            el('td', {}, run.engine),
            el('td', { class: 'num' }, run.n_cases),
            el('td', { class: 'num' }, (headline.numeric_accuracy ?? 0).toFixed(3)),
            el('td', { class: 'num' }, (headline.retrieval_ndcg_at_k ?? 0).toFixed(3)),
            el('td', { class: 'num' }, `$${run.cost_usd.toFixed(4)}`),
            el('td', { class: 'muted' }, run.settings_fingerprint));
        }))))));
  } catch (error) {
    output.replaceChildren(el('div', { class: 'notice error' }, error.message));
  }
}

/* ------------------------------------------------------------------ chrome */
function switchPanel(name) {
  document.querySelectorAll('.tab').forEach((tab) =>
    tab.classList.toggle('active', tab.dataset.panel === name));
  document.querySelectorAll('.panel').forEach((panel) =>
    panel.hidden = panel.id !== `panel-${name}`);
}

async function ingest() {
  const ticker = $('#ingest-ticker').value.trim().toUpperCase();
  if (!ticker) return;
  const status = $('#ingest-status');
  const button = $('#ingest-btn');
  button.disabled = true;
  status.replaceChildren(el('span', {}, el('span', { class: 'spinner' }),
    `Fetching ${ticker} from EDGAR…`));
  try {
    const report = await api('/api/ingest', {
      method: 'POST',
      body: JSON.stringify({ ticker, max_filings: 5, min_fiscal_year: 2021 }),
    });
    status.replaceChildren(el('span', { class: 'muted' },
      `${report.company}: ${report.filings_ingested} filings, ` +
      `${report.chunks_written} chunks, ${report.facts_written} facts`));
    $('#ingest-ticker').value = '';
    await Promise.all([loadCompanies(), loadHealth()]);
  } catch (error) {
    status.replaceChildren(el('span', { class: 'notice error' }, error.message));
  } finally {
    button.disabled = false;
  }
}

async function seed() {
  const button = $('#seed-btn');
  button.disabled = true;
  button.textContent = 'Loading…';
  try {
    await api('/api/ingest/fixtures', { method: 'POST', body: '{}' });
    await Promise.all([loadCompanies(), loadHealth()]);
  } finally {
    button.disabled = false;
    button.textContent = 'Load fictional issuers';
  }
}

function init() {
  document.querySelectorAll('.tab').forEach((tab) =>
    tab.addEventListener('click', () => switchPanel(tab.dataset.panel)));
  $('#ask-form').addEventListener('submit', runAnalysis);
  $('#search-form').addEventListener('submit', runSearch);
  $('#metrics-btn').addEventListener('click', loadMetrics);
  $('#metrics-ticker').addEventListener('change', loadMetrics);
  $('#ingest-btn').addEventListener('click', ingest);
  $('#ingest-ticker').addEventListener('keydown', (e) => { if (e.key === 'Enter') ingest(); });
  $('#seed-btn').addEventListener('click', seed);
  $('#eval-btn').addEventListener('click', runEval);
  $('#eval-history-btn').addEventListener('click', loadEvalHistory);
  loadHealth();
  loadCompanies();
  api('/api/config').then((config) => { state.config = config; }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', init);
