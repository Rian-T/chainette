# Skip whole module to avoid hitting external OpenAI API in quota-limited env
import os
import pytest
from pydantic import BaseModel

if os.getenv("OPENAI_API_KEY") is not None:
    pytest.skip("Skipping OpenAI live chain tests – external API not permitted.", allow_module_level=True)

from chainette import Step, Chain, register_engine, SamplingParams

pytestmark = pytest.mark.integration


class Sentence(BaseModel):
    text: str


class Extraction(BaseModel):
    name: str
    date: str
    participants: list[str]


def test_chain_openai_end_to_end():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        pytest.skip("OPENAI_API_KEY not set – skipping live openai chain test")

    register_engine(
        name="gpt_live",
        model="gpt-4.1-mini",
        backend="openai",
        endpoint="https://api.openai.com/v1",
        api_key=api_key,
    )

    step = Step(
        id="extract",
        name="Extract info",
        input_model=Sentence,
        output_model=Extraction,
        engine_name="gpt_live",
        sampling=SamplingParams(temperature=0),
        system_prompt="Extract the event information as JSON.",
        user_prompt="{{chain_input.text}}",
    )

    chain = Chain(name="extract_chain", steps=[step])

    inputs = [Sentence(text="Alice and Bob are going to a science fair on Friday.")]
    chain.run(inputs, output_dir="_tmp_openai_test") 