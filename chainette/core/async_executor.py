from __future__ import annotations
"""AsyncExecutor – optional async DAG runner (experimental).

Keeps same semantics as `Executor` but exposes an `async run` coroutine so that
clients can integrate Chainette inside asyncio / trio apps seamlessly.

LOC budget intentionally ≤120.
"""
from typing import Any, Dict, List

import anyio
from pydantic import BaseModel

from .graph import Graph
from .step import Step
from .apply import ApplyNode
from .branch import Branch
from ..io.writer import RunWriter
from ..engine.registry import get_engine_config

__all__ = ["AsyncExecutor"]


class AsyncExecutor:  # noqa: D101
    def __init__(self, graph: Graph, batch_size: int = 1):
        self.graph = graph
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    async def run(
        self,
        inputs: List[BaseModel],
        writer: RunWriter,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:  # noqa: D401
        """Asynchronously execute *graph* on *inputs*; return final histories."""
        histories: List[Dict[str, Any]] = [{"chain_input": inp} for inp in inputs]
        nodes = self.graph.nodes()

        active_engine_name: str | None = None
        last_step: Step | None = None

        # Helper to release engine from any thread
        def _release(name: str):
            get_engine_config(name).release_engine()
            if debug:
                print(f"[AsyncExecutor] released engine '{name}'")

        for n in nodes:
            obj = n.ref

            entering_branch = isinstance(obj, Branch)
            upcoming_engine = obj.engine_name if isinstance(obj, Step) else None  # type: ignore[attr-defined]

            if active_engine_name and (entering_branch or (upcoming_engine and upcoming_engine != active_engine_name)):
                await anyio.to_thread.run_sync(_release, active_engine_name)
                if last_step is not None:
                    last_step.engine = None
                active_engine_name = None
                last_step = None

            # -------------------------------------------------- #
            if isinstance(obj, Step):
                new_inputs: List[BaseModel] = []
                new_histories: List[Dict[str, Any]] = []
                bs = self.batch_size if self.batch_size > 0 else len(inputs)
                for s in range(0, len(inputs), bs):
                    e = s + bs
                    batch_in, batch_hist = inputs[s:e], histories[s:e]
                    outs, new_hist = await anyio.to_thread.run_sync(
                        obj.execute, batch_in, batch_hist, writer, debug
                    )
                    new_inputs.extend(outs or batch_in)
                    new_histories.extend(new_hist or batch_hist)
                inputs, histories = new_inputs, new_histories
                active_engine_name = obj.engine_name
                last_step = obj

            elif isinstance(obj, ApplyNode):
                inputs, histories = await anyio.to_thread.run_sync(
                    obj.execute, inputs, histories, writer, debug
                )

            elif isinstance(obj, Branch):
                snapshot_inputs = list(inputs)
                snapshot_hist = [h.copy() for h in histories]
                await anyio.to_thread.run_sync(
                    obj.execute, snapshot_inputs, snapshot_hist, writer, debug
                )

            else:
                raise TypeError(f"Unsupported node type: {type(obj).__name__}")

        if active_engine_name:
            await anyio.to_thread.run_sync(_release, active_engine_name)
            if last_step is not None:
                last_step.engine = None

        return histories 