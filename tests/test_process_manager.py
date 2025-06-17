import types

from chainette.engine.registry import register_engine
from chainette.engine import process_manager

class DummyProc:  # noqa: D401
    def __init__(self):
        self._poll = None
    def poll(self):
        return self._poll
    def terminate(self):
        self._poll = 0
    def wait(self, timeout=None):
        self._poll = 0


def test_ensure_running_no_vllm(monkeypatch):
    # Monkeypatch subprocess.Popen to avoid spawning
    monkeypatch.setattr(process_manager, "subprocess", types.SimpleNamespace(Popen=lambda *a, **k: DummyProc(), DEVNULL=None))
    monkeypatch.setattr(process_manager, "time", types.SimpleNamespace(sleep=lambda x: None))
    import sys
    sys.modules['vllm'] = types.ModuleType('vllm')
    # Register engine without endpoint
    cfg = register_engine("eproc", model="dummy", backend="vllm_api")
    url = process_manager.ensure_running(cfg)
    assert url.startswith("http://localhost:")
    assert cfg.process is not None
    # Second call returns same url & doesn't spawn
    url2 = process_manager.ensure_running(cfg)
    assert url2 == url
    # stop
    process_manager.maybe_stop(cfg)
    assert cfg.process is None 