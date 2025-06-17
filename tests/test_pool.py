import types

import pytest

from chainette.engine.registry import register_engine
from chainette.engine.pool import EnginePool


class DummyLLM:  # noqa: D101
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return "ok"


def test_engine_pool_lru(monkeypatch):
    # Stub VLLMClient so EnginePool doesn't attempt real HTTP calls
    monkeypatch.setattr("chainette.engine.http_client.VLLMClient", DummyLLM)

    register_engine(name="e1", model="dummy-model")
    register_engine(name="e2", model="dummy-model")

    pool = EnginePool(max_size=1)

    eng1 = pool.acquire("e1")
    assert isinstance(eng1, DummyLLM)

    eng2 = pool.acquire("e2")
    assert isinstance(eng2, DummyLLM)
    # LRU capacity 1 → e1 should be evicted
    assert "e1" not in pool._cache

    # e2 still cached
    assert pool._cache["e2"] is eng2

    # Acquiring e1 again loads new instance
    eng1b = pool.acquire("e1")
    assert isinstance(eng1b, DummyLLM)
    assert eng1b is not eng1 