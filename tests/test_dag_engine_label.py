import pytest
from rich.console import Console

from chainette.utils.dag import build_rich_tree, RenderOptions
from chainette.engine.registry import register_engine
from chainette import Chain
from chainette.core.step import Step, SamplingParams
from pydantic import BaseModel


class _In(BaseModel):
    text: str


class _Out(BaseModel):
    summary: str


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dag_contains_engine_label(capsys):
    # Register a dummy engine with an Ollama backend
    register_engine(
        "dummy_eng",
        model="tiny",
        backend="ollama_api",
        endpoint="http://localhost:11434",
    )

    step = Step(
        id="s1",
        name="summarise",
        input_model=_In,
        output_model=_Out,
        engine_name="dummy_eng",
        sampling=SamplingParams(),
    )

    chain = Chain(name="test", steps=[step])

    tree = build_rich_tree(chain, opts=RenderOptions(icons_on=False))
    console = Console(force_terminal=True, width=80)
    console.print(tree)
    out = capsys.readouterr().out

    # Ensure step id and backend label appear
    assert "s1" in out
    assert "(ollama_api)" in out 