"""Shared fixtures.

Every test runs against a temporary database seeded with the synthetic demo
corpus, so the suite is hermetic: no network, no API key, no shared state
between tests.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("CREDITLENS_LOG_LEVEL", "CRITICAL")
os.environ.setdefault("CREDITLENS_LLM_PROVIDER", "offline")


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory) -> Iterator[Path]:
    """A database seeded once with the fixture corpus, reused across tests."""
    directory = tmp_path_factory.mktemp("creditlens-db")
    os.environ["CREDITLENS_DATABASE_URL"] = f"sqlite:///{directory / 'test.db'}"
    os.environ["CREDITLENS_DATA_DIR"] = str(directory)

    from creditlens.config import reset_settings_cache
    from creditlens.db import init_db, reset_engine, session_scope
    from creditlens.ingest.fixtures import load_fixtures

    reset_settings_cache()
    reset_engine()
    init_db(drop=True)
    with session_scope() as session:
        load_fixtures(session)
    yield directory
    reset_engine()


@pytest.fixture
def session(seeded_db):
    from creditlens.db import session_scope

    with session_scope() as db_session:
        yield db_session


@pytest.fixture
def ledger():
    from creditlens.agent.evidence import EvidenceLedger

    return EvidenceLedger()


@pytest.fixture
def tool_ctx(session, ledger):
    from creditlens.agent.tools import ToolContext
    from creditlens.observability import NullTrace

    return ToolContext(session=session, ledger=ledger, trace=NullTrace())


@pytest.fixture
def client(seeded_db):
    from fastapi.testclient import TestClient

    from creditlens.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def period_factory():
    """Build a Period with sensible defaults for ratio tests."""
    import datetime as dt

    from creditlens.finance.statements import Period, add_derived

    def build(fiscal_period: str = "FY", year: int = 2024, **values) -> Period:
        period = Period(
            ticker="TEST", company_id=1, fiscal_year=year,
            fiscal_period=fiscal_period, period_end=dt.date(year, 12, 31),
        )
        base = {
            "revenue": 1000.0, "cost_of_revenue": 600.0, "operating_income": 200.0,
            "depreciation_amortization": 50.0, "interest_expense": 20.0,
            "net_income": 140.0, "pretax_income": 180.0, "income_tax_expense": 40.0,
            "total_assets": 2000.0, "current_assets": 800.0,
            "current_liabilities": 400.0, "inventory": 100.0,
            "short_term_debt": 100.0, "long_term_debt": 500.0,
            "cash_and_equivalents": 200.0, "short_term_investments": 50.0,
            "total_equity": 900.0, "total_liabilities": 1100.0,
            "operating_cash_flow": 250.0, "capex": 80.0,
            "accounts_receivable": 150.0, "retained_earnings": 500.0,
        }
        base.update(values)
        period.values.update({k: v for k, v in base.items() if v is not None})
        add_derived(period)
        return period

    return build
