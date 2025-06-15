from __future__ import annotations
"""JoinBranch – Branch variant that merges its final outputs back into parent history.

LOC < 40. Works by storing *alias* and exposing `execute` that returns updated
histories with `{alias: last_output}` per item. Inputs list is unchanged.
"""
from typing import List, Dict, Any

from pydantic import BaseModel

from .branch import Branch
from ..io.writer import RunWriter

__all__ = ["JoinBranch"]


class JoinBranch(Branch):  # noqa: D101
    def __init__(self, alias: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias = alias

    # ------------------------------------------------------------------ #
    def execute(
        self,
        inputs: List[BaseModel],
        item_histories: List[Dict[str, Any]],
        writer: RunWriter | None = None,
        debug: bool = False,
        batch_size: int = 0,
    ) -> List[Dict[str, Any]]:  # noqa: D401
        # Run as normal inside a copy so we don't mutate items
        branch_inputs = list(inputs)
        branch_histories = [h.copy() for h in item_histories]
        super().execute(branch_inputs, branch_histories, writer, debug, batch_size)

        # Assume each branch produces exactly one output per input (true for Step sequences)
        out_key = self.steps[-1].id if self.steps else None
        for parent_h, br_h in zip(item_histories, branch_histories):
            key = out_key if out_key in br_h else list(br_h.keys())[-1]
            parent_h[self.alias] = br_h[key]
        return item_histories 