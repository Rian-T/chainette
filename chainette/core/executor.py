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
from chainette.utils.events import publish, BatchStarted, BatchFinished, StepTotalItems
from chainette.engine.broker import EngineBroker
from chainette.engine.process_manager import ensure_running, maybe_stop
from chainette.engine.registry import get_engine_config

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
        active_cfg = None  # track current engine cfg for maybe_stop
        if debug:
            print("EXECUTOR order:", [n.id for n in nodes])

        for n in nodes:
            obj = n.ref

            # -------------------------------------------------- #
            # Execute node types
            # -------------------------------------------------- #
            if isinstance(obj, Step):
                # Ensure required engine process is up
                cfg = get_engine_config(obj.engine_name)

                if cfg is not active_cfg:
                    # switching engines – maybe stop previous lazy engine
                    if active_cfg is not None:
                        maybe_stop(active_cfg)
                    ensure_running(cfg)
                    active_cfg = cfg

                new_inputs: List[BaseModel] = []
                new_histories: List[Dict[str, Any]] = []

                # Announce total items once for accurate progress UI
                if len(inputs) > 0:
                    total_batches = (len(inputs) + self.batch_size - 1) // self.batch_size
                    publish(StepTotalItems(step_id=obj.id, total=total_batches))

                # Run in mini-batches managed here (Step ignores batch_size arg)
                bs = self.batch_size if self.batch_size > 0 else len(inputs)
                batch_no = 0
                for start in range(0, len(inputs), bs):
                    end = start + bs
                    batch_inp = inputs[start:end]
                    batch_hist = histories[start:end]

                    publish(BatchStarted(step_id=obj.id, batch_no=batch_no, count=len(batch_inp)))

                    outs, hist_out = obj.execute(batch_inp, batch_hist, writer, debug=debug)

                    publish(BatchFinished(step_id=obj.id, batch_no=batch_no, count=1))

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

        EngineBroker.flush(force=True)

        if active_cfg is not None:
            maybe_stop(active_cfg)

        return histories

    # ------------------------------------------------------------------ #
    def run_iter(
        self,
        inputs: List[BaseModel],
        writer: RunWriter,
        debug: bool = False,
    ):
        """Yield dictionaries for each processed batch.

        Emitted structure: {"step_id": str, "batch_no": int, "outputs": list, "histories": list}
        Consumers may ignore or store these as needed.  This keeps memory low
        for very large datasets while still allowing downstream streaming.
        """

        histories: List[Dict[str, Any]] = [
            {"chain_input": inp} for inp in inputs
        ]

        nodes = self.graph.nodes()
        active_cfg = None  # track current engine cfg for maybe_stop
        if debug:
            print("EXECUTOR order:", [n.id for n in nodes])

        for n in nodes:
            obj = n.ref

            # -------------------------------------------------- #
            # Execute node types
            # -------------------------------------------------- #
            if isinstance(obj, Step):
                # Ensure required engine process is up
                cfg = get_engine_config(obj.engine_name)

                if cfg is not active_cfg:
                    # switching engines – maybe stop previous lazy engine
                    if active_cfg is not None:
                        maybe_stop(active_cfg)
                    ensure_running(cfg)
                    active_cfg = cfg

                # Announce total items once
                if len(inputs) > 0:
                    total_batches = (len(inputs) + self.batch_size - 1) // self.batch_size
                    publish(StepTotalItems(step_id=obj.id, total=total_batches))

                bs = self.batch_size if self.batch_size > 0 else len(inputs)
                batch_no = 0
                new_inputs: List[BaseModel] = []
                new_histories: List[Dict[str, Any]] = []

                for start in range(0, len(inputs), bs):
                    end = start + bs
                    batch_inp = inputs[start:end]
                    batch_hist = histories[start:end]

                    publish(BatchStarted(step_id=obj.id, batch_no=batch_no, count=len(batch_inp)))

                    outs, hist_out = obj.execute(batch_inp, batch_hist, writer, debug=debug)

                    publish(BatchFinished(step_id=obj.id, batch_no=batch_no, count=1))

                    if outs:
                        new_inputs.extend(outs)
                        new_histories.extend(hist_out)
                    else:
                        new_inputs.extend(batch_inp)
                        new_histories.extend(batch_hist)

                    yield {
                        "step_id": obj.id,
                        "batch_no": batch_no,
                        "outputs": outs,
                        "histories": hist_out,
                    }

                    # encourage GC
                    del batch_inp, batch_hist, outs, hist_out

                    batch_no += 1

                inputs = new_inputs
                histories = new_histories

            else:
                # Fallback to original logic for non-Step
                inputs, histories = obj.execute(inputs, histories, writer, debug=debug)

        yield {
            "step_id": "_end_",
            "batch_no": -1,
            "outputs": inputs,
            "histories": histories,
        }

        if active_cfg is not None:
            maybe_stop(active_cfg) 