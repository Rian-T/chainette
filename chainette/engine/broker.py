from __future__ import annotations
"""EngineBroker – single entry-point for acquiring / flushing engines.

Implements a tiny reference-count mechanism on top of the global
`ENGINE_POOL`.  Use::

    from chainette.engine.broker import EngineBroker as EB
    with EB.acquire("gemma_ollama") as eng:
        eng.generate(...)

`flush(force=True)` releases engines that are idle (ref_count==0) or all
when *force* is True.
"""

from contextlib import contextmanager
from time import time
from typing import Iterator

from .pool import ENGINE_POOL
from .registry import get_engine_config

_ID_IDLE_SEC = 180  # engine evicted if idle > this value


class _Tracker:  # noqa: D401
    __slots__ = ("engine", "ref", "last")

    def __init__(self, engine):
        self.engine = engine
        self.ref = 0
        self.last = time()


class _BrokerImpl:  # noqa: D401
    def __init__(self):
        self._track: dict[str, _Tracker] = {}

    # ------------------------------------------------------------------ #
    @contextmanager
    def acquire(self, name: str) -> Iterator:
        """Context-manager yielding live engine for *name*."""
        eng = ENGINE_POOL.acquire(name)
        t = self._track.setdefault(name, _Tracker(eng))
        t.ref += 1
        try:
            yield eng
        finally:
            t.ref -= 1
            t.last = time()

    # ------------------------------------------------------------------ #
    def flush(self, *, force: bool = False):  # noqa: D401
        """Release engines idle for > idle_sec or everything when *force*."""
        now = time()
        for name in list(self._track.keys()):
            tr = self._track[name]
            if force or (tr.ref == 0 and now - tr.last > _ID_IDLE_SEC):
                get_engine_config(name).release_engine()
                self._track.pop(name, None)
                ENGINE_POOL.pop(name)


EngineBroker = _BrokerImpl() 