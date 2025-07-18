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
        return [SimpleNamespace(outputs=[SimpleNamespace(text='{"result":"ok"}')])]

@pytest.mark.parametrize("n", [3])
def test_parallel_branches_refcount(monkeypatch, n):
    # prepare dummy engines & configs
    engines = {f"e{i}": DummyEng(f"e{i}") for i in range(n)}
    import chainette.engine.registry as reg
    reg._REGISTRY.clear()
    for name in engines:
        eng = engines[name]
        reg._REGISTRY[name] = SimpleNamespace(engine=eng, release_engine=lambda e=eng: setattr(e, "released", True), model="dummy")
    # patch tokenizer
    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return ""
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: _Tok()))

    chain_steps = []
    for i in range(n):
        st = Step(
            id=f"s{i}",
            name=f"s{i}",
            output_model=Out,
            engine_name=f"e{i}",
            sampling=SamplingParams(),
            user_prompt="dummy",
        )
        chain_steps.append(st)

    chain = Chain(name="par", steps=chain_steps)
    inp = [Dummy()]
    chain.run(inp, output_dir="_tmp_parallel", writer=None, debug=False)

    # after run engines should be released
    EngineBroker.flush(force=True)
    for name,e in engines.items():
        print(name, getattr(e,"released",False))
    assert all(getattr(e, "released", False) for e in engines.values()) 