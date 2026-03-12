"""
Structured logging setup using structlog.

Provides JSON logs in production, pretty console logs in development.
Includes metric counters for key operational events.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import defaultdict
from typing import Any

import structlog

_configured = False
_lock = threading.Lock()


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog and stdlib logging once."""
    global _configured
    with _lock:
        if _configured:
            return
        _configured = True

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    from app.api.log_broadcaster import log_broadcaster

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        log_broadcaster,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name)


class MetricsCounter:
    """Thread-safe counters for operational metrics."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def increment(self, metric: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[metric] += amount

    def get(self, metric: str) -> int:
        with self._lock:
            return self._counts[metric]

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


# Global metrics instance
metrics = MetricsCounter()
