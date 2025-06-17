from __future__ import annotations

"""Chain orchestration for Chainette."""

from pathlib import Path
from typing import List, Union, Dict, Any

from pydantic import BaseModel

from chainette.core.node import Node
from chainette.core.branch import Branch
from chainette.io.writer import RunWriter

# New executor-based runtime
from .graph import Graph, GraphNode
from .executor import Executor

__all__ = ["Chain"]


class Chain:
    def __init__(
        self,
        *,
        name: str,
        steps: List[Union[Node, List[Branch]]], # A step can now also be a single Branch
        emoji: str | None = None,
        batch_size: int = 1,
    ) -> None:
        self.name = name
        self.steps = steps
        self.emoji = emoji or ""
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #

    def run(
        self,
        inputs: List[BaseModel],
        *,
        writer: RunWriter | Any | None = None,  # StreamWriter or RunWriter
        output_dir: str | Path | None = None,
        fmt: str = "jsonl",
        generate_flattened_output: bool = True,
        max_lines_per_file: int = 1000,
        debug: bool = False,
        show_ui: bool = True,
    ):
        """Execute the chain and write results.

        A custom *writer* (e.g. StreamWriter) can be provided.  If absent, a
        legacy `RunWriter` is created using *output_dir*.
        """

        if writer is None:
            if output_dir is None:
                raise ValueError("output_dir must be provided when writer is None")

            writer = RunWriter(Path(output_dir), max_lines_per_file=max_lines_per_file, fmt=fmt)

        # In both cases set chain name if attribute exists
        if hasattr(writer, "set_chain_name"):
            writer.set_chain_name(self.name)

        # ---- Build Graph from linear steps list ---------------------------------
        graph_nodes: List[GraphNode] = []
        prev_gn: GraphNode | None = None

        for item in self.steps:
            if isinstance(item, list):
                # list of branches – connect previous node to each branch root
                for br in item:
                    gn = GraphNode(id=br.id, ref=br)
                    if prev_gn is not None:
                        prev_gn.connect(gn)
                    graph_nodes.append(gn)
                # branches don't become new prev_gn (main flow continues)
            else:
                gn = GraphNode(id=item.id, ref=item)
                if prev_gn is not None:
                    prev_gn.connect(gn)
                else:
                    graph_nodes.append(gn)  # root node
                prev_gn = gn

        if not graph_nodes:
            raise ValueError("Chain has no executable nodes.")

        graph = Graph(roots=graph_nodes[:1])  # first root; downstream links set above

        if show_ui:
            try:
                from chainette.utils.banner import ChainetteBanner
                from chainette.utils.logging_v2 import show_dag_tree

                ChainetteBanner().display()  # type: ignore[call-arg]
                step_ids = [s.id if not isinstance(s, list) else "parallel_branches" for s in self.steps]
                show_dag_tree(step_ids)
            except Exception:  # noqa: BLE001
                # UI is optional – ignore failures (e.g., non-TTY env)
                pass

        executor = Executor(graph, batch_size=self.batch_size)

        # ------------------------------------------------------------------ #
        histories = executor.run(inputs, writer, debug=debug)

        # Finalise / close writer
        if isinstance(writer, RunWriter):
            return writer.finalize(generate_flattened_output=generate_flattened_output)
        else:
            # StreamWriter just needs to be closed
            if hasattr(writer, "close"):
                writer.close()
            return None
