from __future__ import annotations

"""chainette.core.branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A Branch is merely a named list of chain nodes executed in order.
"""

from dataclasses import dataclass
from typing import List, Sequence, Union # Added Union

from pydantic import BaseModel

from chainette.core.apply import ApplyNode
from chainette.core.step import Step

Node = Union[Step, ApplyNode]  # Use Union for type hint

__all__ = ["Branch", "Node"]


@dataclass
class Branch:  # noqa: D401
    name: str
    steps: Sequence[Node]
    emoji: str = "🌿" # Added default emoji

    def run(self, inputs: List[BaseModel], run_id: str, branch_index: int) -> List[Any]:  # noqa: D401
        """Execute all nodes in the branch sequentially."""
        data: List[Any] = inputs
        for step_index, node in enumerate(self.steps):
            # Pass run_id and a combined index (branch_index, step_index) if needed
            # For simplicity, passing run_id and the step's index within the branch
            data = node.run(data, run_id=run_id, step_index=step_index)
        return data
