from __future__ import annotations
"""Rich live logger (v3) – accurate progress using StepTotalItems.

This keeps the public helpers identical to v2 so callers can switch
imports seamlessly. LOC ≤ 120.
"""
from collections import defaultdict
from typing import Dict, Any

from rich.console import Console
from rich.tree import Tree
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from chainette.utils.events import (
    subscribe,
    BatchFinished,
    StepTotalItems,
)
from chainette.utils.dag import build_rich_tree

console = Console()

# --------------------------------------------------------------------------- #
# Progress handling
# --------------------------------------------------------------------------- #
_progress: Progress | None = None
_tasks: Dict[str, int] = {}
_counts: Dict[str, int] = defaultdict(int)


def _ensure_progress() -> Progress:  # noqa: D401
    global _progress
    if _progress is None:
        _progress = Progress(
            TextColumn("[bold blue]{task.fields[id]}[/]"),
            BarColumn(),
            "{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        _progress.start()
    return _progress


# --------------------------------------------------------------------------- #
# Event subscribers
# --------------------------------------------------------------------------- #
@subscribe(StepTotalItems)
def _on_total(evt: StepTotalItems):  # noqa: D401 – event hook
    prog = _ensure_progress()
    if evt.step_id not in _tasks:
        _tasks[evt.step_id] = prog.add_task(
            description="", total=evt.total, id=evt.step_id  # type: ignore[arg-type]
        )
    else:
        prog.update(_tasks[evt.step_id], total=evt.total)


@subscribe(BatchFinished)
def _on_batch_finish(evt: BatchFinished):  # noqa: D401 – event hook
    prog = _ensure_progress()
    task_id = _tasks.get(evt.step_id)
    if task_id is not None:
        prog.update(task_id, advance=evt.count)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def show_dag_tree(obj: Any, **kw):  # noqa: D401
    """Render the execution DAG tree (Chain or legacy list)."""
    if hasattr(obj, "steps"):
        try:
            console.print(build_rich_tree(obj, **kw))
            return
        except Exception:
            pass
    # legacy list fallback
    tree = Tree("[bold]Execution DAG[/]")
    for sid in obj:
        tree.add(f"[cyan]{sid}[/]")
    console.print(tree)


def stop():  # noqa: D401
    global _progress
    if _progress is not None:
        _progress.stop()
        _progress = None 