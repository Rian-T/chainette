"""Pure Python apply() demo (placeholder)."""

from pydantic import BaseModel
from chainette import Chain
from chainette.core.apply import ApplyNode

class X(BaseModel):
    x: int

class Double(BaseModel):
    y: int

def double(inp: X):  # noqa: D401
    return [Double(y=inp.x * 2)]

double_step = ApplyNode(double, id="double", input_model=X)

py_chain = Chain(name="Pure Python Demo", steps=[double_step]) 