from __future__ import annotations
"""Minimal JoinBranch example using pure Python `apply` nodes.

Run:
    poetry run chainette run examples/join/inc_dec_join.py inc_dec_chain inputs.jsonl _tmp_run_join

This requires no LLM engine – all logic is pure Python, keeping the example
small and runnable everywhere.
"""
from typing import List
from pydantic import BaseModel, Field

from chainette import Chain, Branch, apply

# Models ------------------------------------------------------------------- #

class Number(BaseModel):
    value: int = Field(..., description="input number")


# Apply helpers ------------------------------------------------------------ #

def inc(n: Number) -> List[Number]:
    """Increment by 1."""
    return [Number(value=n.value + 1)]

def dec(n: Number) -> List[Number]:
    """Decrement by 1."""
    return [Number(value=n.value - 1)]

inc_node = apply(inc, input_model=Number, output_model=Number)  # type: ignore[arg-type]
dec_node = apply(dec, input_model=Number, output_model=Number)  # type: ignore[arg-type]

# Branches with join ------------------------------------------------------- #

inc_branch = Branch(name="inc_branch", steps=[inc_node]).join("inc")
dec_branch = Branch(name="dec_branch", steps=[dec_node]).join("dec")

# Chain -------------------------------------------------------------------- #

inc_dec_chain = Chain(
    name="Increment/Decrement Join Demo",
    steps=[[inc_branch, dec_branch]],  # parallel then join
    batch_size=4,
)

# ------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Quick self-test when run directly
    inputs = [Number(value=v) for v in (3, 7)]
    inc_dec_chain.run(inputs, output_dir="_tmp_run_join") 