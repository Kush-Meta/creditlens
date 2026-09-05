"""Engine/session management and schema bootstrap."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from creditlens.config import get_settings
from creditlens.db.migrations import apply_additive_migrations
from creditlens.db.models import Base
from creditlens.observability import get_logger

log = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _configure_sqlite(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    # WAL + normal sync: concurrent readers during ingestion without fsync cost.
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is None:
        url = get_settings().resolved_database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, connect_args=connect_args, pool_pre_ping=True)
        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _configure_sqlite)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        log.info("database engine created", extra={"url": _redact(url)})
    return _engine


def _redact(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


def init_db(drop: bool = False) -> None:
    engine = get_engine()
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    migrated = apply_additive_migrations(engine)
    log.info("schema ready", extra={
        "tables": len(Base.metadata.tables), "dropped": drop,
        "migrated_columns": migrated or None,
    })


def session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    factory = session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


def reset_engine() -> None:
    """Drop cached engine - used by tests that repoint the database URL."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


def healthcheck() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        log.exception("database healthcheck failed")
        return False
