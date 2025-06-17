import os
import pytest
import httpx
from pydantic import BaseModel

from chainette.engine.registry import register_engine
from chainette.core.step import Step, SamplingParams
from chainette.core.chain import Chain

pytestmark = pytest.mark.integration

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def _server_up() -> bool:
    try:
        httpx.get(OLLAMA_URL + "/api/tags", timeout=2.0)
        return True
    except Exception:
        return False


class Sentence(BaseModel):
    text: str


class Echo(BaseModel):
    echo: str


@pytest.mark.skipif(not _server_up(), reason="Ollama server not running")
@pytest.mark.parametrize("lazy_flag", [True, False])
def test_single_engine_lazy_variants(lazy_flag):
    name = f"gemma_{'lazy' if lazy_flag else 'eager'}"
    register_engine(
        name=name,
        model="gemma3:1b",
        backend="ollama_api",
        endpoint=OLLAMA_URL,
        lazy=lazy_flag,
    )

    step = Step(
        id="echo",
        name="Echo",
        input_model=Sentence,
        output_model=Echo,
        engine_name=name,
        sampling=SamplingParams(temperature=0.0),
        system_prompt="Echo the user text in JSON with key 'echo'.",
        user_prompt="{{chain_input.text}}",
    )
    chain = Chain(name="echo-chain", steps=[step])

    chain.run([Sentence(text="hello world")], output_dir="_tmp_ollama_single")


@pytest.mark.skipif(not _server_up(), reason="Ollama server not running")
def test_multiple_engines_switch():
    register_engine(
        name="gemma_a",
        model="gemma3:1b",
        backend="ollama_api",
        endpoint=OLLAMA_URL,
    )
    register_engine(
        name="gemma_b",
        model="gemma3:1b",
        backend="ollama_api",
        endpoint=OLLAMA_URL,
    )

    step1 = Step(
        id="s1",
        name="First",
        input_model=Sentence,
        output_model=Echo,
        engine_name="gemma_a",
        sampling=SamplingParams(temperature=0.0),
        system_prompt="Respond with JSON {\"echo\": <text>}.",
        user_prompt="{{chain_input.text}}",
    )

    chain = Chain(name="double-echo", steps=[step1])
    chain.run([Sentence(text="ping")], output_dir="_tmp_ollama_multi") 