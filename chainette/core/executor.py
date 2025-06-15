from __future__ import annotations
"""Generic DAG executor for Chainette (experimental).

Uses the new `Graph` utilities to traverse and execute nodes while
handling per-item histories, batching and engine reuse.  Current version
focuses on sequential depth-first execution to keep LOC low.
"""
from typing import Any, Dict, List

from pydantic import BaseModel

from .graph import Graph
from .step import Step
from .apply import ApplyNode
from .branch import Branch
from .join_branch import JoinBranch
from ..io.writer import RunWriter
from ..engine.registry import get_engine_config
from chainette.utils.events import publish, BatchStarted, BatchFinished

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
        histories: List[Dict[str, Any]] = [{"chain_input": inp} for inp in inputs]
        nodes = self.graph.nodes()
        if debug:
            print("EXECUTOR order:", [n.id for n in nodes])

        active_engine_name: str | None = None
        last_step_obj: Step | None = None

        for n in nodes:
            obj = n.ref

            # -------------------------------------------------- #
            # Engine life-cycle: release when switching context
            # -------------------------------------------------- #
            entering_branch = isinstance(obj, Branch)
            upcoming_engine = obj.engine_name if isinstance(obj, Step) else None  # type: ignore[attr-defined]

            if active_engine_name and (entering_branch or (upcoming_engine and upcoming_engine != active_engine_name)):
                if debug:
                    reason = "branch" if entering_branch else f"engine switch → {upcoming_engine}"
                    print(f"Releasing engine '{active_engine_name}' due to {reason}.")
                get_engine_config(active_engine_name).release_engine()
                if last_step_obj is not None:
                    last_step_obj.engine = None  # pyright: ignore[reportGeneralTypeIssues]
                active_engine_name = None
                last_step_obj = None

            # -------------------------------------------------- #
            # Execute node types
            # -------------------------------------------------- #
            if isinstance(obj, Step):
                new_inputs: List[BaseModel] = []
                new_histories: List[Dict[str, Any]] = []

                # Run in mini-batches managed here (Step ignores batch_size arg)
                bs = self.batch_size if self.batch_size > 0 else len(inputs)
                batch_no = 0
                for start in range(0, len(inputs), bs):
                    end = start + bs
                    batch_inp = inputs[start:end]
                    batch_hist = histories[start:end]

                    publish(BatchStarted(step_id=obj.id, batch_no=batch_no, count=len(batch_inp)))

                    outs, hist_out = obj.execute(batch_inp, batch_hist, writer, debug=debug)

                    publish(BatchFinished(step_id=obj.id, batch_no=batch_no, count=len(outs)))

                    # Fallback: when parsing fails, Step may return fewer outputs.
                    # In that case, keep original inputs/histories aligned.
                    if outs:
                        new_inputs.extend(outs)
                        new_histories.extend(hist_out)
                    else:
                        new_inputs.extend(batch_inp)
                        new_histories.extend(batch_hist)

                    batch_no += 1

                inputs = new_inputs
                histories = new_histories

                active_engine_name = obj.engine_name
                last_step_obj = obj

            elif isinstance(obj, ApplyNode):
                inputs, histories = obj.execute(inputs, histories, writer, debug=debug)

            elif isinstance(obj, JoinBranch):
                # Execute and merge histories
                histories = obj.execute(inputs, histories, writer, debug=debug)

            elif isinstance(obj, Branch):
                # Regular branch – fire-and-forget
                snapshot_inputs = list(inputs)
                snapshot_histories = [h.copy() for h in histories]
                obj.execute(snapshot_inputs, snapshot_histories, writer, debug=debug)

            else:
                raise TypeError(f"Unsupported node type: {type(obj).__name__}")

        # Release engine at end of run
        if active_engine_name:
            if debug:
                print(f"Final release of engine '{active_engine_name}'.")
            get_engine_config(active_engine_name).release_engine()
            if last_step_obj is not None:
                last_step_obj.engine = None  # pyright: ignore[reportGeneralTypeIssues]

        return histories 