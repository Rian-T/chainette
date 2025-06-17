import time
from types import SimpleNamespace

from chainette.engine.broker import EngineBroker
from chainette.engine.engine_pool import ENGINE_POOL


class DummyEngine:
    def __init__(self):
        self.gen_count = 0

    def generate(self, *a, **kw):
        self.gen_count += 1
        return ["ok"]


class DummyCfg(SimpleNamespace):
    def release_engine(self):
        self.released = True


def test_ref_count_and_flush(monkeypatch):
    eng = DummyEngine()
    cfg = DummyCfg(engine=eng, released=False)

    # ensure clean state
    EngineBroker._track.clear()
    monkeypatch.setattr("chainette.engine.registry.get_engine_config", lambda n: cfg)
    monkeypatch.setattr("chainette.engine.engine_pool.get_engine_config", lambda n: cfg)
    monkeypatch.setattr("chainette.engine.broker.get_engine_config", lambda n: cfg)

    # acquire context should keep ref>0 so flush shouldn't release
    with EngineBroker.acquire("dummy") as e:
        assert e is eng
        EngineBroker.flush()
        assert cfg.released is False

    # now ref==0 → flush should release
    EngineBroker.flush(force=True)
    assert cfg.released is True 