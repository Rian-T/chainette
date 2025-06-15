from pydantic import BaseModel
from chainette import Step, Branch, SamplingParams, Chain
from chainette.dsl import pipe


class A(BaseModel):
    foo: str

class B(BaseModel):
    bar: str

s1 = Step(id="s1", name="s1", input_model=A, output_model=B, engine_name="e", sampling=SamplingParams())
s2 = Step(id="s2", name="s2", input_model=B, output_model=B, engine_name="e", sampling=SamplingParams())
branch = Branch(name="b", steps=[s2])


def test_dsl_to_chain():
    ch: Chain = (pipe(s1) >> pipe(branch)).to_chain("dsl")
    assert len(ch.steps) == 2
    assert isinstance(ch.steps[1], Branch)


def test_operator_syntax():
    chain = (s1 >> branch).to_chain("op")
    assert isinstance(chain, Chain) 