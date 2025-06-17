import os
import pytest
import httpx
from pydantic import BaseModel

from chainette.engine.registry import register_engine
from chainette.core.step import Step, SamplingParams
from chainette.core.chain import Chain

pytestmark = pytest.mark.integration


VLLM_URL = os.getenv("VLLM_API_URL", "http://localhost:8000")


def _server_available() -> bool:
    try:
        httpx.get(VLLM_URL + "/health", timeout=2.0)
        return True
    except Exception:
        return False


class Sentence(BaseModel):
    text: str


class Extraction(BaseModel):
    name: str
    date: str
    participants: list[str]


def test_chain_vllm_end_to_end():
    if not _server_available():
        pytest.skip("vLLM serve not reachable – set VLLM_API_URL or run api_server")

    register_engine(
        name="vllm_live",
        model="gpt-4o-2024-08-06",  # any model loaded in server
        backend="vllm_api",
        endpoint=VLLM_URL + "/v1",
    )

    step = Step(
        id="extract",
        name="Extract info",
        input_model=Sentence,
        output_model=Extraction,
        engine_name="vllm_live",
        sampling=SamplingParams(temperature=0),
        system_prompt="Extract the event information as JSON.",
        user_prompt="{{chain_input.text}}",
    )
    chain = Chain(name="extract-vllm", steps=[step])

    chain.run([Sentence(text="Alice and Bob are going to a science fair on Friday.")], output_dir="_tmp_vllm_test") 