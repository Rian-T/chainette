from chainette.engine.broker import EngineBroker
from chainette.core.step import Step, SamplingParams
from chainette.core.branch import Branch
from chainette.core.chain import Chain
from pydantic import BaseModel
from types import SimpleNamespace
import pytest

class Dummy(BaseModel):
    text: str = "x"

class Out(BaseModel):
    result: str = "y"

class DummyEng:  # simple counter
    def __init__(self, name: str):
        self.name = name
        self.calls = 0
    def generate(self, *a, **k):
        self.calls += 1
        return ["{\"result\":\"ok\"}"]

@pytest.mark.parametrize("n", [3])
def test_parallel_branches_refcount(monkeypatch, n):
    # prepare dummy engines & configs
    engines = {f"e{i}": DummyEng(f"e{i}") for i in range(n)}
    def _cfg(name):
        eng = engines[name]
        return SimpleNamespace(engine=eng, release_engine=lambda: setattr(eng, "released", True))
    monkeypatch.setattr("chainette.engine.registry.get_engine_config", _cfg)
    monkeypatch.setattr("chainette.engine.engine_pool.get_engine_config", _cfg)
    monkeypatch.setattr("chainette.engine.broker.get_engine_config", _cfg)

    steps = []
    for i in range(n):
        st = Step(
            id=f"s{i}",
            name=f"s{i}",
            output_model=Out,
            engine_name=f"e{i}",
            sampling=SamplingParams(),
        )
        steps.append(Branch(name=f"b{i}", steps=[st]))

    chain = Chain(name="par", steps=[steps])
    inp = [Dummy()]
    chain.run(inp, output_dir="_tmp_parallel", writer=None, debug=False)

    # after run engines should be released
    EngineBroker.flush(force=True)
    assert all(getattr(e, "released", False) for e in engines.values()) 