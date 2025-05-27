from __future__ import annotations

"""Base Node class for the Chainette execution graph."""

from typing import List, Tuple, Dict, Any

from pydantic import BaseModel

# Forward declaration for RunWriter to avoid circular import if Node needs it directly
# However, it's passed as an argument, so not strictly necessary here yet.
# from chainette.io.writer import RunWriter # Uncomment if type hint needed in Node itself

__all__ = ["Node"]


class Node:  # noqa: D101 – minimalist base class
    id: str
    name: str

    def execute(
        self,
        inputs: List[BaseModel],
        item_histories: List[Dict[str, Any]],
        writer: 'RunWriter | None' = None,
        debug: bool = False,
        batch_size: int = 0,  # Added batch_size parameter
    ) -> Tuple[List[BaseModel], List[Dict[str, Any]]]:
        """Run the node and return (outputs, updated_item_histories)."""
        raise NotImplementedError