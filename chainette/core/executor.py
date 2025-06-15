from __future__ import annotations
"""Generic DAG executor for Chainette (experimental).

Uses the new `Graph` utilities to traverse and execute nodes while
handling per-item histories, batching and engine reuse.  Current version
focuses on sequential depth-first execution to keep LOC low.
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .graph import Graph, GraphNode
from .step import Step
from .apply import ApplyNode
from .branch import Branch
from ..io.writer import RunWriter

__all__ = ["Executor"]


class Executor:  # noqa: D101
    def __init__(self, graph: Graph, batch_size: int = 1):
        self.graph = graph
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    def run(
        self,
        inputs: List[BaseModel],
        writer: RunWriter,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute *graph* on *inputs*; return final histories."""
        # Histories parallel to current item list
        histories: List[Dict[str, Any]] = [{"chain_input": inp} for inp in inputs]
        nodes = self.graph.nodes()
        if debug:
            print("EXECUTOR order:", [n.id for n in nodes])

        for n in nodes:
            obj = n.ref
            if isinstance(obj, Step):
                outputs, histories = obj.execute(
                    inputs, histories, writer, debug=debug, batch_size=self.batch_size
                )
                inputs = outputs if outputs else inputs  # keep flow when Apply returns []
            elif isinstance(obj, ApplyNode):
                inputs, histories = obj.execute(inputs, histories, writer, debug=debug)
            elif isinstance(obj, Branch):
                # Branch executes but does not modify main flow
                obj.execute(inputs, histories, writer, debug=debug)
            else:
                raise TypeError(f"Unsupported node type: {type(obj).__name__}")
        return histories 