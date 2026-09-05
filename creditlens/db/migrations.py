"""Additive schema migrations.

`Base.metadata.create_all` creates missing tables but never alters existing
ones, so a column added after a database exists is silently absent until
something queries it. This applies additive column migrations idempotently at
startup, which is the right weight for a single-service project.

It is deliberately additive only - no drops, no type changes, no data
rewrites. Anything beyond that should move to Alembic rather than grow here.
"""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from creditlens.observability import get_logger

log = get_logger(__name__)

#: table -> column -> DDL type. Every entry must be nullable or defaulted,
#: because it is applied to tables that already hold rows.
ADDITIVE_COLUMNS: dict[str, dict[str, str]] = {
    "companies": {
        "sector": "VARCHAR(64)",
        "profile": "VARCHAR(32)",
    },
}


def apply_additive_migrations(engine: Engine) -> list[str]:
    """Add any declared column that the live schema is missing."""
    inspector = inspect(engine)
    applied: list[str] = []

    with engine.begin() as connection:
        for table, columns in ADDITIVE_COLUMNS.items():
            if not inspector.has_table(table):
                continue  # create_all will build it with every column present
            existing = {c["name"] for c in inspector.get_columns(table)}
            for column, ddl_type in columns.items():
                if column in existing:
                    continue
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                )
                applied.append(f"{table}.{column}")

    if applied:
        log.info("additive migrations applied", extra={"columns": applied})
    return applied
