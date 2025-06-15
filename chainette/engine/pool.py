from __future__ import annotations
"""EnginePool – simple LRU cache for engine instances.
Keeps up to *max_size* engines alive; evicts least-recently-used calling
`EngineConfig.release_engine()`.
"""
from collections import OrderedDict
from typing import Dict, Any

from .registry import get_engine_config

__all__ = ["EnginePool", "ENGINE_POOL"]


class EnginePool:  # noqa: D101
    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    # -------------------------------------------------- #
    def acquire(self, name: str):
        """Return live engine for *name*, loading if necessary."""
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        cfg = get_engine_config(name)
        eng = cfg.engine
        self._cache[name] = eng
        if len(self._cache) > self.max_size:
            old_name, _ = self._cache.popitem(last=False)
            get_engine_config(old_name).release_engine()
        return eng

    # -------------------------------------------------- #
    def release_all(self):
        """Release all cached engines."""
        for n in list(self._cache.keys()):
            get_engine_config(n).release_engine()
        self._cache.clear()


ENGINE_POOL = EnginePool() 