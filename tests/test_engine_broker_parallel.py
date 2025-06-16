from chainette.engine.broker import EngineBroker
from chainette.core.step import Step, SamplingParams
from chainette.core.branch import Branch
from chainette.core.chain import Chain
from pydantic import BaseModel
from types import SimpleNamespace
import pytest
import transformers

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
    import chainette.engine.registry as reg
    reg._REGISTRY.clear()
    for name in engines:
        reg._REGISTRY[name] = SimpleNamespace(engine=engines[name], release_engine=lambda: setattr(engines[name], "released", True), model="dummy")
    # patch tokenizer
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: None))

    steps = []
    for i in range(n):
        st = Step(
            id=f"s{i}",
            name=f"s{i}",
            output_model=Out,
            engine_name=f"e{i}",
            sampling=SamplingParams(),
            user_prompt="dummy",
        )
        steps.append(Branch(name=f"b{i}", steps=[st]))

    chain = Chain(name="par", steps=[steps])
    inp = [Dummy()]
    chain.run(inp, output_dir="_tmp_parallel", writer=None, debug=False)

    # after run engines should be released
    EngineBroker.flush(force=True)
    assert all(getattr(e, "released", False) for e in engines.values()) 