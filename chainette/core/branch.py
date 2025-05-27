from __future__ import annotations

"""Branching support for Chainette.

A Branch is simply a named sequence of nodes executed with the same input.
Outputs are *not* merged back into the main chain – use separate branches at
chain tail like in the examples.
"""

from typing import List, Tuple, Dict, Any, Optional

from pydantic import BaseModel

from chainette.io.writer import RunWriter
from .node import Node
from .step import Step

__all__ = ["Branch"]


class Branch(Node):  # noqa: D101
    def __init__(self, name: str, steps: List[Node]):
        self.id = name
        self.name = name
        self.steps = steps

    # -------------------------------------------------- #

    def execute(
        self, 
        inputs: List[BaseModel], 
        item_histories: List[Dict[str, Any]], 
        writer: RunWriter | None = None,
        debug: bool = False,
        batch_size: int = 0, # Added batch_size parameter
    ) -> None:
        if writer is not None:
            writer.add_node_to_graph({"id": self.id, "name": self.name, "type": "Branch"})

        current_branch_inputs = list(inputs)
        current_branch_histories = [hist.copy() for hist in item_histories]

        if debug:
            print(f"\nDEBUG: Branch '{self.name}' executing with {len(current_branch_inputs)} items.")

        for node_idx, node in enumerate(self.steps):
            if debug:
                node_type = type(node).__name__
                print(f"  DEBUG (Branch '{self.name}'): Running {node_type}: {getattr(node, 'name', 'Unknown')} ({node_idx+1}/{len(self.steps)})")
                if isinstance(node, Step) and current_branch_inputs and current_branch_histories:
                    sample_prompt = node._build_prompt(current_branch_histories[0])
                    print(f"    DEBUG (Branch '{self.name}'): Sample prompt for first item:\n    {'-'*30}\n    {sample_prompt}\n    {'-'*30}")
            
            current_branch_inputs, current_branch_histories = node.execute(
                current_branch_inputs, current_branch_histories, writer, debug=debug, batch_size=batch_size # Pass batch_size
            )
            
            if debug:
                print(f"  DEBUG (Branch '{self.name}'): Node {getattr(node, 'name', 'Unknown')} finished. {len(current_branch_inputs)} items produced.")
        
        if debug:
            print(f"DEBUG: Branch '{self.name}' execution finished.")