import os
import pytest
import httpx
from pydantic import BaseModel

from chainette.engine.registry import register_engine
from chainette.core.step import Step, SamplingParams
from chainette.core.chain import Chain

pytestmark = pytest.mark.integration

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def _ollama_running() -> bool:
    try:
        httpx.get(OLLAMA_URL + "/api/tags", timeout=2.0)
        return True
    except Exception:
        return False


class Sentence(BaseModel):
    text: str


class Extraction(BaseModel):
    name: str
    date: str
    participants: list[str]


def test_chain_ollama_end_to_end():
    if not _ollama_running():
        pytest.skip("Ollama server not running on localhost:11434")

    register_engine(
        name="ollama_live",
        model="qwen2.5-instruct",  # ensure model pulled
        backend="ollama_api",
        endpoint=OLLAMA_URL,
    )

    step = Step(
        id="extract",
        name="Extract info",
        input_model=Sentence,
        output_model=Extraction,
        engine_name="ollama_live",
        sampling=SamplingParams(temperature=0.1),
        system_prompt="Extract the event information as JSON.",
        user_prompt="{{chain_input.text}}",
    )
    chain = Chain(name="extract-ollama", steps=[step])

    chain.run([Sentence(text="Alice and Bob are going to a science fair on Friday.")], output_dir="_tmp_ollama_test") 