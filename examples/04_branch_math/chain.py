"""Branching + Join example (placeholder)."""

from pydantic import BaseModel
from chainette import Chain, Branch
from chainette.core.apply import ApplyNode

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class Number(BaseModel):
    value: int

class Result(BaseModel):
    new_value: int

# --------------------------------------------------------------------------- #
# Pure python functions
# --------------------------------------------------------------------------- #

def inc(x: Number):  # noqa: D401
    return [Result(new_value=x.value + 1)]

def dec(x: Number):  # noqa: D401
    return [Result(new_value=x.value - 1)]

inc_step = ApplyNode(inc, id="inc", input_model=Number)
dec_step = ApplyNode(dec, id="dec", input_model=Number)

inc_branch = Branch(name="inc_branch", steps=[inc_step]).join("inc")
dec_branch = Branch(name="dec_branch", steps=[dec_step]).join("dec")

branching_chain = Chain(name="Branching Demo", steps=[[inc_branch, dec_branch]]) 