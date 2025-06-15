from pydantic import BaseModel
from chainette import Step, Branch, SamplingParams
from chainette.dsl import pipe
from chainette.core.join_branch import JoinBranch


class A(BaseModel):
    x: str

class B(BaseModel):
    y: str

step1 = Step(id="s1", name="s1", input_model=A, output_model=B, engine_name="e", sampling=SamplingParams())
step2 = Step(id="s2", name="s2", input_model=B, output_model=B, engine_name="e", sampling=SamplingParams())
branch = Branch(name="br", steps=[step2]).join("alias")


def test_join_branch_type():
    assert isinstance(branch, JoinBranch)


def test_pipe_join():
    chain = (pipe(step1) >> (pipe(branch))).to_chain("join")
    assert len(chain.steps) == 2 