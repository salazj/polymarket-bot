"""
Log broadcaster — captures structlog events and streams them to WebSocket clients.

Integrates as a structlog processor so every log event is intercepted
without modifying any existing logging call sites.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any


class LogEntry:
    """Lightweight log entry for the ring buffer."""

    __slots__ = ("timestamp", "level", "event", "logger", "data")

    def __init__(
        self,
        timestamp: str,
        level: str,
        event: str,
        logger: str,
        data: dict[str, Any],
    ) -> None:
        self.timestamp = timestamp
        self.level = level
        self.event = event
        self.logger = logger
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "event": self.event,
            "logger": self.logger,
            "data": self.data,
        }


class LogBroadcaster:
    """
    Structlog processor that captures log events and broadcasts them.

    Add to the structlog processor chain. Call ``get_recent()`` for the REST
    endpoint, and register WebSocket queues with ``add_subscriber()`` /
    ``remove_subscriber()``.
    """

    def __init__(self, buffer_size: int = 5000) -> None:
        self._buffer: deque[LogEntry] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue[LogEntry]] = set()
        self._sub_lock = threading.Lock()

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Structlog processor hook — intercept every log event."""
        entry = LogEntry(
            timestamp=event_dict.get("timestamp", ""),
            level=method_name,
            event=str(event_dict.get("event", "")),
            logger=event_dict.get("logger", ""),
            data={
                k: _safe_serialize(v)
                for k, v in event_dict.items()
                if k not in ("timestamp", "event", "logger", "level", "log_level")
            },
        )

        with self._lock:
            self._buffer.append(entry)

        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass

        return event_dict

    def get_recent(
        self, limit: int = 200, level_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Return recent log entries, optionally filtered by level."""
        with self._lock:
            entries = list(self._buffer)

        if level_filter:
            _levels = _level_set(level_filter)
            entries = [e for e in entries if e.level in _levels]

        return [e.to_dict() for e in entries[-limit:]]

    def add_subscriber(self) -> asyncio.Queue[LogEntry]:
        """Register a new WebSocket subscriber and return its queue."""
        q: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=500)
        with self._sub_lock:
            self._subscribers.add(q)
        return q

    def remove_subscriber(self, q: asyncio.Queue[LogEntry]) -> None:
        with self._sub_lock:
            self._subscribers.discard(q)


# Global singleton
log_broadcaster = LogBroadcaster()


def _safe_serialize(v: Any) -> Any:
    """Coerce values to JSON-safe types."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def _level_set(min_level: str) -> set[str]:
    """Return the set of levels at or above ``min_level``."""
    hierarchy = ["debug", "info", "warning", "error", "critical"]
    min_level = min_level.lower()
    try:
        idx = hierarchy.index(min_level)
    except ValueError:
        return set(hierarchy)
    return set(hierarchy[idx:])
