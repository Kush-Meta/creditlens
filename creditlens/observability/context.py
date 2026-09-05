"""Ambient run/span context, propagated with contextvars.

Kept in its own module so logging can read it without importing tracing
(which imports logging).
"""
from __future__ import annotations

from contextvars import ContextVar

_run_id: ContextVar[str | None] = ContextVar("creditlens_run_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("creditlens_span_id", default=None)


def current_run_id() -> str | None:
    return _run_id.get()


def current_span_id() -> str | None:
    return _span_id.get()


def set_run_id(run_id: str | None):
    return _run_id.set(run_id)


def set_span_id(span_id: str | None):
    return _span_id.set(span_id)


def reset_run_id(token) -> None:
    _run_id.reset(token)


def reset_span_id(token) -> None:
    _span_id.reset(token)
