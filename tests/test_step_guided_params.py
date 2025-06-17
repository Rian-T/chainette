from pydantic import BaseModel
from chainette.engine.registry import register_engine
from chainette.core.step import Step, SamplingParams


class DummyIn(BaseModel):
    text: str


class DummyOut(BaseModel):
    summary: str


def test_guided_params_disabled_for_openai():
    register_engine("gpt_test", model="gpt-4.1-mini", backend="openai")
    step = Step(
        id="s1",
        name="test",
        input_model=DummyIn,
        output_model=DummyOut,
        engine_name="gpt_test",
        sampling=SamplingParams(),
    )
    assert getattr(step.sampling, "guided_decoding", None) is None


def test_guided_params_enabled_for_vllm_api():
    register_engine("vserve", model="qwen1.5-1b", backend="vllm_api", endpoint="http://dummy:8000/v1")
    step = Step(
        id="s2",
        name="test2",
        input_model=DummyIn,
        output_model=DummyOut,
        engine_name="vserve",
        sampling=SamplingParams(),
    )
    assert step.sampling.guided_decoding is not None 