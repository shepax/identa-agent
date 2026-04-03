"""In-memory LRU cache for calibration results.

AUDIT-FIX: 1.5 — Replaces the dead-code stub that returned None on every get()
and silently swallowed put() calls. Uses functools.lru_cache-style eviction via
an OrderedDict so the engine can avoid redundant API calls within a session.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from identa.calibration.types import CalibrationResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 256


class CalibrationCache:
    """Bounded in-memory cache for calibration results (LRU eviction)."""

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE):
        """
        Args:
            max_size: Maximum number of results to keep. Oldest entries
                      are evicted when the cache exceeds this size.
        """
        self._store: OrderedDict[str, CalibrationResult] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> CalibrationResult | None:
        """Return the cached result for *key*, or None if not present."""
        if key not in self._store:
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        logger.debug(f"Cache hit for key={key!r}")
        return self._store[key]

    def put(self, key: str, result: CalibrationResult) -> None:
        """Store *result* under *key*, evicting the LRU entry if full."""
        self._store[key] = result
        self._store.move_to_end(key)
        if len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug(f"Cache evicted LRU key={evicted_key!r}")

    def __len__(self) -> int:
        return len(self._store)
