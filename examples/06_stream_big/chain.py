"""Streaming writer demo placeholder."""

from pydantic import BaseModel
from chainette import Chain
from chainette.core.apply import ApplyNode

class Num(BaseModel):
    n: int

class Square(BaseModel):
    sq: int

def square(x: Num):  # noqa: D401
    return [Square(sq=x.n * x.n)]

square_step = ApplyNode(square, id="square", input_model=Num)

stream_chain = Chain(name="Streaming Demo", steps=[square_step]) 