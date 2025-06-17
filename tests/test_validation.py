from pydantic import BaseModel
from chainette import Step, Chain, SamplingParams
from chainette.utils.validate import validate_chain_io


class A(BaseModel):
    x: str

class B(BaseModel):
    y: str


valid_step = Step(
    id="s1",
    name="s1",
    input_model=A,
    output_model=B,
    engine_name="e",
    sampling=SamplingParams(),
)

a_chain = Chain(name="ok", steps=[valid_step])


def test_validate_ok():
    assert validate_chain_io(a_chain) == []


def test_validate_mismatch():
    bad_step = Step(
        id="s2",
        name="s2",
        input_model=A,  # expects A but will get B
        output_model=A,
        engine_name="e",
        sampling=SamplingParams(),
    )
    chain = Chain(name="bad", steps=[valid_step, bad_step])
    mism = validate_chain_io(chain)
    assert len(mism) == 1
    node_id, expected, actual = mism[0]
    assert node_id == "s2" and expected == B and actual == A 