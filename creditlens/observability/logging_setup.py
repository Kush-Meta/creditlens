"""Structured logging.

Every log line carries the ambient trace/run id so a single analysis can be
reconstructed end-to-end from stdout alone (the only thing you reliably have
in a container).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from creditlens.observability.context import current_run_id, current_span_id

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        run_id = current_run_id()
        if run_id:
            payload["run_id"] = run_id
        span_id = current_span_id()
        if span_id:
            payload["span_id"] = span_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _safe(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        run_id = current_run_id()
        prefix = f"[{run_id[:8]}] " if run_id else ""
        extras = {
            k: _safe(v)
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        tail = " " + json.dumps(extras, default=str) if extras else ""
        base = f"{record.levelname:<7} {prefix}{record.name}: {record.getMessage()}{tail}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value][:50]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in list(value.items())[:50]}
    return str(value)


_configured = False


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    global _configured
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        from creditlens.config import get_settings

        settings = get_settings()
        configure_logging(settings.log_level, settings.log_format)
    return logging.getLogger(name)
