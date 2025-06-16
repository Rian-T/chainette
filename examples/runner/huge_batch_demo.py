from __future__ import annotations
"""Huge batch demo – processes 1 000 000 dummy records with Apply nodes.

Run:
    poetry run chainette run examples/runner/huge_batch_demo.py demo_chain inputs_huge.jsonl out_huge --stream-writer --quiet

The Apply nodes simply increment and double numbers to keep runtime/GPU
requirements low.  Demonstrates runner scalability + StreamWriter.
"""

from typing import List
from pathlib import Path
import json

from pydantic import BaseModel

from chainette import apply, Chain, ApplyNode

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class Number(BaseModel):
    value: int

class NumberOut(BaseModel):
    value: int


# --------------------------------------------------------------------------- #
# Pure-python transforms (fast)
# --------------------------------------------------------------------------- #

def inc(x: Number) -> List[NumberOut]:
    return [NumberOut(value=x.value + 1)]

def double(x: NumberOut) -> List[NumberOut]:
    return [NumberOut(value=x.value * 2)]


inc_node: ApplyNode = apply(inc, input_model=Number, output_model=NumberOut)
double_node: ApplyNode = apply(double, input_model=NumberOut, output_model=NumberOut)


# --------------------------------------------------------------------------- #
# Chain definition
# --------------------------------------------------------------------------- #

demo_chain = Chain(name="Huge Batch Demo", steps=[inc_node, double_node], batch_size=10000)


# --------------------------------------------------------------------------- #
# Helper to generate large input file quickly (only when missing)
# --------------------------------------------------------------------------- #

def _ensure_inputs(path: Path, n: int = 1_000_000):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"value": i}) + "\n")


if __name__ == "__main__":
    input_path = Path("inputs_huge.jsonl")
    _ensure_inputs(input_path)
    print("Input file generated at", input_path) 