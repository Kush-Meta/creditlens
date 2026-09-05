"""Command-line interface.

Every operation the service performs is reachable from here, so the system can
be driven, measured and debugged without the HTTP layer.
"""
from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from creditlens import __version__
from creditlens.config import get_settings
from creditlens.observability import configure_logging

app = typer.Typer(
    name="creditlens",
    help="Evidence-grounded credit analysis of public companies.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _setup(log_level: str = "INFO") -> None:
    settings = get_settings()
    configure_logging(log_level, settings.log_format)
    from creditlens.db import init_db

    init_db()


@app.command()
def version() -> None:
    """Print version and effective configuration fingerprint."""
    settings = get_settings()
    console.print(f"CreditLens {__version__}")
    console.print(f"provider={settings.llm_provider} model={settings.model or '(provider default)'}")
    console.print(f"database={settings.resolved_database_url}")
    console.print(f"settings fingerprint={settings.fingerprint()}")


@app.command()
def providers() -> None:
    """List supported LLM providers and which credentials resolve here."""
    _setup("WARNING")
    from creditlens.agent.llm import available_providers, probe_engine

    active = probe_engine()
    table = Table(title="LLM providers")
    for column in ("provider", "default model", "credentials", "env"):
        table.add_column(column)
    for entry in available_providers():
        marker = "[green]detected[/green]" if entry["credentials_detected"] else "[dim]-[/dim]"
        table.add_row(
            entry["provider"], entry["default_model"], marker,
            ", ".join(entry["credential_env"]) or entry["notes"][:40],
        )
    console.print(table)
    console.print(
        f"\nactive: [bold]{active.get('provider')}[/bold] "
        f"model={active.get('model')} reachable={active.get('reachable')}"
    )
    if not active.get("reachable"):
        console.print("[dim]falling back to the deterministic engine; answers are labelled as such[/dim]")


@app.command()
def ingest(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker to ingest from SEC EDGAR."),
    max_filings: int = typer.Option(5, help="Number of recent filings to parse."),
    min_year: int | None = typer.Option(None, help="Drop facts before this fiscal year."),
    log_level: str = typer.Option("INFO", help="Log level."),
) -> None:
    """Ingest one issuer's XBRL facts and filing text from SEC EDGAR."""
    _setup(log_level)
    from creditlens.db import session_scope
    from creditlens.ingest.pipeline import ingest_ticker

    with session_scope() as session:
        try:
            report = ingest_ticker(
                session, ticker, max_filings=max_filings, min_fiscal_year=min_year
            )
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(report.to_dict()))


@app.command()
def seed(
    log_level: str = typer.Option("INFO", help="Log level."),
) -> None:
    """Load the offline demo corpus (four fictional issuers, synthetic data)."""
    _setup(log_level)
    from creditlens.db import session_scope
    from creditlens.ingest.fixtures import load_fixtures

    with session_scope() as session:
        reports = load_fixtures(session)
    table = Table(title="Demo corpus loaded (synthetic, fictional issuers)")
    for column in ("ticker", "company", "filings", "chunks", "facts"):
        table.add_column(column)
    for report in reports:
        table.add_row(report.ticker, report.company_name,
                      str(report.filings_ingested), str(report.chunks_written),
                      str(report.facts_written))
    console.print(table)


@app.command()
def universe(
    max_filings: int = typer.Option(12, help="Filings to parse per issuer."),
    min_year: int = typer.Option(2021, help="Drop facts before this fiscal year."),
    workers: int = typer.Option(4, help="Parallel EDGAR fetchers (the shared rate limiter still applies)."),
    sector: str | None = typer.Option(None, help="Restrict to one sector."),
    profile: str | None = typer.Option(None, help="Restrict to one sampling profile."),
    fresh: bool = typer.Option(False, help="Re-ingest issuers that are already present."),
    log_level: str = typer.Option("WARNING", help="Log level."),
) -> None:
    """Ingest the curated issuer universe from SEC EDGAR."""
    _setup(log_level)
    from creditlens.db import session_scope
    from creditlens.ingest import universe as universe_mod
    from creditlens.ingest.pipeline import ingest_universe

    issuers = [
        i for i in universe_mod.UNIVERSE
        if (sector is None or i.sector == sector)
        and (profile is None or i.profile == profile)
    ]
    if not issuers:
        console.print("[red]no issuers match those filters[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"Ingesting {len(issuers)} issuers, up to {max_filings} filings each, "
        f"{workers} parallel fetchers.\n"
    )
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console,
    ) as progress_bar:
        task = progress_bar.add_task("fetching", total=len(issuers))

        def on_progress(done: int, total: int, ticker: str, report) -> None:
            detail = (
                f"{ticker}: {report.filings_ingested} filings, {report.chunks_written} chunks"
                if report else f"{ticker}: failed"
            )
            progress_bar.update(task, completed=done, description=detail)

        result = ingest_universe(
            session_scope, issuers,
            max_filings=max_filings, min_fiscal_year=min_year,
            workers=workers, resume=not fresh, progress=on_progress,
        )

    totals = result.totals
    table = Table(title="Universe ingest")
    for column in ("metric", "value"):
        table.add_column(column)
    table.add_row("issuers ingested", str(result.ingested))
    table.add_row("already present (skipped)", str(len(result.skipped)))
    table.add_row("failed", str(len(result.failed)))
    table.add_row("filings", f"{totals['filings']:,}")
    table.add_row("chunks", f"{totals['chunks']:,}")
    table.add_row("financial facts", f"{totals['facts']:,}")
    table.add_row("duration", f"{result.duration_s:.0f}s")
    console.print(table)
    for failure in result.failed:
        console.print(f"[yellow]failed:[/yellow] {failure}")


@app.command()
def reembed(
    provider: str | None = typer.Option(None, help="Embedding provider: hashed, ollama, openai, google."),
    model: str | None = typer.Option(None, help="Embedding model name."),
    dim: int | None = typer.Option(None, help="Vector dimension."),
    batch_size: int = typer.Option(64, help="Chunks per embedding request."),
    limit: int | None = typer.Option(None, help="Stop after this many chunks (for trials)."),
    log_level: str = typer.Option("WARNING", help="Log level."),
) -> None:
    """Rebuild stored vectors with a different embedding model."""
    _setup(log_level)
    from creditlens.db import session_scope
    from creditlens.retrieval.embeddings import build_embedder
    from creditlens.retrieval.reembed import embedding_inventory
    from creditlens.retrieval.reembed import reembed as run_reembed

    embedder = build_embedder(provider, model, dim)
    console.print(f"target model: [bold]{embedder.name}[/bold] (dim {embedder.dim})")

    with session_scope() as session:
        before = embedding_inventory(session)
    console.print(f"stored now: {before['embedded']:,} vectors across "
                  f"{len(before['by_model'])} model(s)\n")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    ) as bar:
        task = bar.add_task("embedding", total=None)

        def on_progress(done: int, total: int, rate: float) -> None:
            bar.update(task, completed=done, total=total,
                       description=f"{rate:.1f} chunks/s")

        with session_scope() as session:
            report = run_reembed(session, embedder=embedder, batch_size=batch_size,
                                 limit=limit, progress=on_progress)

    table = Table(title="Re-embed")
    for column in ("metric", "value"):
        table.add_column(column)
    for key, value in report.to_dict().items():
        if key != "errors":
            table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)
    for error in report.errors:
        console.print(f"[yellow]{error}[/yellow]")

    with session_scope() as session:
        after = embedding_inventory(session)
    for entry in after["by_model"]:
        console.print(f"  [dim]{entry['model']} (dim {entry['dim']}): {entry['chunks']:,} chunks[/dim]")
    if after["unembedded"]:
        console.print(f"[yellow]{after['unembedded']:,} chunk(s) still unembedded[/yellow]")


@app.command()
def companies() -> None:
    """List issuers held in the corpus."""
    _setup("WARNING")
    from creditlens.agent.evidence import EvidenceLedger
    from creditlens.agent.tools import ToolContext, tool_list_companies
    from creditlens.db import session_scope
    from creditlens.observability import NullTrace

    with session_scope() as session:
        payload = tool_list_companies(
            ToolContext(session=session, ledger=EvidenceLedger(), trace=NullTrace())
        )
    table = Table(title="Corpus")
    for column in ("ticker", "name", "source", "latest period", "periods"):
        table.add_column(column)
    for company in payload["companies"]:
        table.add_row(
            company["ticker"], company["name"][:34], company["data_source"],
            str(company["latest_period"]), str(len(company["periods_available"])),
        )
    console.print(table)


@app.command()
def ratios(
    ticker: str = typer.Argument(..., help="Ticker."),
    period: str = typer.Option("TTM", help="latest | TTM | FY2024 | Q2-2025."),
) -> None:
    """Compute the full ratio set for one issuer-period."""
    _setup("WARNING")
    from creditlens.agent.evidence import EvidenceLedger
    from creditlens.agent.tools import ToolContext, ToolError, tool_compute_ratios
    from creditlens.db import session_scope
    from creditlens.observability import NullTrace

    with session_scope() as session:
        try:
            payload = tool_compute_ratios(
                ToolContext(session=session, ledger=EvidenceLedger(), trace=NullTrace()),
                ticker=ticker, period=period,
            )
        except ToolError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    table = Table(title=f"{payload['ticker']} {payload['period']} ratios")
    for column in ("ratio", "value", "formula", "note"):
        table.add_column(column)
    for row in payload["ratios"]:
        table.add_row(
            row["label"], row["display"], row["formula"],
            row["unavailable_reason"] or "",
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The credit question."),
    ticker: list[str] | None = typer.Option(None, "--ticker", "-t", help="Scope to issuer(s)."),
    provider: str | None = typer.Option(None, help="Pin a provider (anthropic, openai, google, ollama, ...)."),
    model: str | None = typer.Option(None, help="Override the model."),
    offline: bool = typer.Option(False, help="Force the deterministic engine."),
    json_out: bool = typer.Option(False, "--json", help="Emit the raw response."),
    log_level: str = typer.Option("WARNING", help="Log level."),
) -> None:
    """Run one analysis."""
    _setup(log_level)
    from creditlens.agent.llm import OfflineEngine, build_engine
    from creditlens.agent.orchestrator import Orchestrator
    from creditlens.db import session_scope

    engine = OfflineEngine() if offline else build_engine(provider, model)
    with session_scope() as session:
        result = Orchestrator(session, engine=engine).analyze(
            question, tickers=[t.upper() for t in ticker] if ticker else None
        )

    if json_out:
        console.print_json(json.dumps(result.to_dict(include_trace=False), default=str))
        return

    console.rule(f"[bold]{question}")
    console.print(f"[bold]Direction:[/bold] {result.credit_direction}   "
                  f"[bold]Engine:[/bold] {result.engine}/{result.model}   "
                  f"[bold]Confidence:[/bold] {result.verification['confidence']}")
    console.print(f"\n{result.answer}\n")

    if result.key_metrics:
        table = Table(title="Key metrics")
        for column in ("metric", "value", "period", "ticker"):
            table.add_column(column)
        for metric in result.key_metrics:
            table.add_row(str(metric.get("name")), str(metric.get("value")),
                          str(metric.get("period")), str(metric.get("ticker", "")))
        console.print(table)

    for title, items in (("Positive factors", result.positive_factors),
                         ("Risk factors", result.risk_factors),
                         ("Caveats", result.caveats)):
        if items:
            console.print(f"\n[bold]{title}[/bold]")
            for item in items:
                console.print(f"  - {item}")

    if result.reasoning:
        console.print(f"\n[bold]Reasoning[/bold]\n{result.reasoning}")

    if result.citations:
        console.print("\n[bold]Citations[/bold]")
        for citation in result.citations:
            console.print(
                f"  [{citation['label']}] {citation['ticker']} {citation['source']} "
                f"item {citation['item']} ({citation['filing_date']})"
            )

    verification = result.verification
    console.print(
        f"\n[dim]verified {verification['numeric_claims_verified']}/"
        f"{verification['numeric_claims_checked']} figures, "
        f"unsupported rate {verification['unsupported_claim_rate']:.0%}, "
        f"citation validity {verification['citation_validity']:.0%}, "
        f"{len(result.tool_calls)} tool calls, {result.latency_ms:.0f} ms, "
        f"${result.cost_usd:.4f}[/dim]"
    )
    if result.degraded_reason:
        console.print(f"[yellow]{result.degraded_reason}[/yellow]")


@app.command("gen-eval")
def gen_eval(
    source: str = typer.Option(
        "universe", help="Issuer source: 'universe' (real, needs an ingest) or 'fixtures'."
    ),
    name: str | None = typer.Option(None, help="Suite name to write. Defaults to the source."),
    rotating: int = typer.Option(2, help="Rotating question families per issuer."),
    validate: bool = typer.Option(True, help="Drop cases the corpus cannot score."),
) -> None:
    """Generate the universe evaluation suite from the curated issuer list."""
    _setup("WARNING")
    from creditlens.db import session_scope
    from creditlens.eval.generate import SOURCES, build_suite, validate_suite, write_suite

    if source not in SOURCES:
        console.print(f"[red]unknown source {source!r}; choose from {', '.join(SOURCES)}[/red]")
        raise typer.Exit(code=1)
    name = name or source
    cases = build_suite(SOURCES[source](), rotating_per_issuer=rotating)
    dropped: list[str] = []
    if validate:
        with session_scope() as session:
            cases, dropped = validate_suite(session, cases)
    path = write_suite(cases, name=name)

    from collections import Counter

    table = Table(title=f"Generated suite: {name}")
    for column in ("dimension", "breakdown"):
        table.add_column(column)
    for label, counts in (
        ("category", Counter(c.category for c in cases)),
        ("sampling profile", Counter(c.profile for c in cases)),
        ("sector", Counter(c.sector for c in cases)),
    ):
        table.add_row(label, ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    table.add_row("total cases", str(len(cases)))
    table.add_row("written to", str(path))
    console.print(table)
    if dropped:
        console.print(f"\n[yellow]{len(dropped)} case(s) dropped as unscoreable[/yellow]")
        for reason in dropped[:12]:
            console.print(f"  [dim]{reason}[/dim]")
        if len(dropped) > 12:
            console.print(f"  [dim]... and {len(dropped) - 12} more[/dim]")


@app.command("eval")
def run_eval(
    suite: str = typer.Option(
        "golden", help="Suite name, or several comma-separated ('golden,universe')."
    ),
    limit: int | None = typer.Option(None, help="Run only the first N cases."),
    provider: str | None = typer.Option(None, help="Pin a provider for this run."),
    model: str | None = typer.Option(None, help="Override the model."),
    offline: bool = typer.Option(False, help="Force the deterministic engine."),
    ablation: bool = typer.Option(False, help="Also run the retrieval ablation."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw metrics."),
) -> None:
    """Run the evaluation suite and print the metric table."""
    _setup("WARNING")
    from creditlens.agent.llm import OfflineEngine, build_engine
    from creditlens.db import session_scope
    from creditlens.eval.runner import run_retrieval_ablation, run_suite

    engine = OfflineEngine() if offline else build_engine(provider, model)
    with session_scope() as session:
        result = run_suite(session, suite=suite, limit=limit, engine=engine)
        ablation_result = run_retrieval_ablation(session, suite=suite) if ablation else None

    if json_out:
        payload = result.to_dict()
        if ablation_result:
            payload["ablation"] = ablation_result
        console.print_json(json.dumps(payload, default=str))
        return

    headline = result.metrics["headline"]
    table = Table(title=f"{suite} - {result.n_cases} cases - engine {result.engine}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in headline.items():
        table.add_row(key, "n/a" if value is None else f"{value}")
    console.print(table)

    slices = result.metrics.get("slices", {})
    for dimension, title in (("by_profile", "By sampling profile"),
                             ("by_sector", "By sector"),
                             ("by_origin", "By case origin")):
        rows = slices.get(dimension) or {}
        if len(rows) < 2:
            continue
        slice_table = Table(title=title)
        for column in ("slice", "cases", "numeric", "unsupported", "nDCG@k", "tool F1"):
            slice_table.add_column(column, justify="right" if column != "slice" else "left")
        for name, entry in rows.items():
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
            slice_table.add_row(
                name, str(entry["cases"]), fmt(entry["numeric_accuracy"]),
                fmt(entry["unsupported_claim_rate"]), fmt(entry["retrieval_ndcg_at_k"]),
                fmt(entry["tool_selection_f1"]),
            )
        console.print(slice_table)

    failing = [r for r in result.results if r.failures]
    if failing:
        console.print(f"\n[yellow]{len(failing)} case(s) with findings[/yellow]")
        for case in failing:
            for failure in case.failures:
                console.print(f"  [dim]{case.case_id}[/dim] {failure}")
    if result.skipped:
        console.print(f"\n[dim]skipped: {'; '.join(result.skipped)}[/dim]")

    if ablation_result:
        ablation_table = Table(title="Retrieval ablation")
        for column in ("configuration", "recall_n", "precision@k", "MRR", "nDCG@k"):
            ablation_table.add_column(column)
        for name, metrics in ablation_result["configurations"].items():
            ablation_table.add_row(
                name,
                *[f"{metrics[key]:.4f}" if metrics[key] is not None else "n/a"
                  for key in ("recall_normalized", "precision_at_k", "mrr", "ndcg_at_k")],
            )
        console.print(ablation_table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    ticker: list[str] | None = typer.Option(None, "--ticker", "-t"),
    item: list[str] | None = typer.Option(None, "--item", help="Filing item, e.g. 1A."),
    top_k: int = typer.Option(6, help="Results to return."),
) -> None:
    """Hybrid search over ingested filing text."""
    _setup("WARNING")
    from creditlens.db import session_scope
    from creditlens.retrieval import hybrid

    with session_scope() as session:
        result = hybrid.search(
            session, query,
            tickers=[t.upper() for t in ticker] if ticker else None,
            item_codes=item, top_k=top_k,
        )
    console.print(f"[dim]{result.diagnostics}[/dim]\n")
    for hit in result.chunks:
        citation = hit.record.citation()
        console.print(
            f"[bold]{hit.rank}. {citation['ticker']} {citation['form_type']} "
            f"{citation['period']} item {citation['item']}[/bold] "
            f"[dim](score {hit.score:.4f})[/dim]"
        )
        console.print(f"   {hit.snippet}\n")


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host."),
    port: int | None = typer.Option(None, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes."),
) -> None:
    """Run the HTTP API and web UI."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "creditlens.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
