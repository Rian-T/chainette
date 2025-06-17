from __future__ import annotations
"""Advanced EnginePool with last-used timestamps (≤70 LOC).
Used internally by EngineBroker; external API: `acquire(name)` → engine.
"""
from collections import OrderedDict
from time import time
from typing import Any

from .registry import get_engine_config

__all__ = ["ENGINE_POOL"]

_MAX_SIZE = 4  # keep a few engines hot


class _Entry:  # noqa: D401
    __slots__ = ("engine", "last")

    def __init__(self, engine: Any):
        self.engine = engine
        self.last = time()


class EnginePool:  # noqa: D101
    def __init__(self, max_size: int = _MAX_SIZE):
        self.max_size = max_size
        self._lru: "OrderedDict[str, _Entry]" = OrderedDict()
        # Simple dict mapping name -> engine for backward-compat
        self._cache: dict[str, Any] = {}

    # -------------------------------------------------- #
    def acquire(self, name: str):
        if name in self._lru:
            entry = self._lru.pop(name)
            entry.last = time()
            self._lru[name] = entry  # move to end
            self._cache[name] = entry.engine
            return entry.engine

        cfg = get_engine_config(name)
        eng = cfg.engine  # lazy-load happens inside registry

        self._lru[name] = _Entry(eng)
        self._cache[name] = eng
        if len(self._lru) > self.max_size:
            old_name, _ = self._lru.popitem(last=False)
            self._cache.pop(old_name, None)
            get_engine_config(old_name).release_engine()
        return eng

    # -------------------------------------------------- #
    def pop(self, name: str):  # helper for Broker.flush
        self._lru.pop(name, None)
        self._cache.pop(name, None)


ENGINE_POOL = EnginePool() 