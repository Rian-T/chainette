from __future__ import annotations

"""Apply nodes integrate plain Python callables into a Chain.

They receive the list of outputs from the previous node and must return a
list that will become the next node's input.
"""

from typing import Callable, Generic, List, Sequence, TypeVar, Type, Optional, Dict, Any
from pydantic import BaseModel

from chainette.core.node import Node
from chainette.io.writer import RunWriter

# Generic type vars for stronger typing
IN = TypeVar("IN", bound=BaseModel)
OUT = TypeVar("OUT", bound=BaseModel)

__all__ = ["ApplyNode", "apply", "Apply"]


class ApplyNode(Node, Generic[IN, OUT]):  # noqa: D101
    def __init__(
        self,
        fn: Callable[[IN], List[OUT]],
        *,
        id: str | None = None,
        name: str | None = None,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None
    ):
        self.fn = fn
        self.id = id or fn.__name__
        self.name = name or fn.__name__.replace("_", " ").title()
        self.input_model = input_model
        self.output_model = output_model

    # -------------------------------------------------- #

    def execute(
        self,
        inputs: List[IN],
        item_histories: List[Dict[str, Any]],
        writer: RunWriter | None = None,
        debug: bool = False,
        batch_size: int = 0,  # Added batch_size, though ApplyNode typically processes item by item
    ) -> tuple[List[OUT], List[Dict[str, Any]]]:
        all_outputs: List[OUT] = []
        new_item_histories: List[Dict[str, Any]] = []
        
        if debug:
            print(f"\n{'='*80}\nAPPLY NODE: {self.id}\n{'='*80}")
            if item_histories:
                print(f"\nITEM HISTORY (first item):\n{'-'*40}")
                for k, v in item_histories[0].items():
                    print(f"{k}: {v}")
                print(f"{'-'*40}")

        for i, input_item in enumerate(inputs):
            current_input_history = item_histories[i]
            outputs_for_this_item: List[OUT] = self.fn(input_item)

            for out_obj in outputs_for_this_item:
                all_outputs.append(out_obj)
                h = current_input_history.copy()
                h[self.id] = out_obj
                new_item_histories.append(h)

        if writer is not None:
            writer.add_node_to_graph({"id": self.id, "name": self.name, "type": "Apply"})
            writer.write_step(self.id, all_outputs)

        return all_outputs, new_item_histories


# Convenience helpers ------------------------------------------------------- #

def apply(
    fn: Callable[[IN], List[OUT]],
    *,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None
) -> ApplyNode[IN, OUT]:  # noqa: D401
    """Return an :class:`ApplyNode` from *fn*."""
    return ApplyNode(fn=fn, input_model=input_model, output_model=output_model)


# Backwards-compat alias (the docs/examples import `Apply` directly)
Apply = ApplyNode