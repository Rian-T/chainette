from __future__ import annotations
"""Rich live logger (v2) for Chainette Runner.

LOC ≤ 80.
Subscribes to EventBus events to show a tree of the DAG and live
progress bars per Step.
"""
from collections import defaultdict
from typing import Dict

from rich.console import Console
from rich.tree import Tree
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from chainette.utils.events import subscribe, BatchStarted, BatchFinished

console = Console()

# Active progress instance and task mapping
_progress: Progress | None = None
_tasks: Dict[str, int] = {}
_counts: Dict[str, int] = defaultdict(int)


def _ensure_progress() -> Progress:
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
# Event handlers
# --------------------------------------------------------------------------- #
@subscribe(BatchStarted)
def _on_batch_start(evt: BatchStarted):  # noqa: D401 – event hook
    prog = _ensure_progress()
    if evt.step_id not in _tasks:
        _tasks[evt.step_id] = prog.add_task(
            description="", total=0, id=evt.step_id  # type: ignore[arg-type]
        )
    task_id = _tasks[evt.step_id]
    # Dynamic total: grow as we discover more items
    prog.update(task_id, total=prog.tasks[task_id].total + evt.count)


@subscribe(BatchFinished)
def _on_batch_finish(evt: BatchFinished):  # noqa: D401 – event hook
    prog = _ensure_progress()
    task_id = _tasks.get(evt.step_id)
    if task_id is not None:
        prog.update(task_id, advance=evt.count)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def show_dag_tree(step_ids):  # noqa: D401
    """Render a simple tree of *step_ids* before execution."""
    tree = Tree("[bold]Execution DAG[/]")
    for sid in step_ids:
        tree.add(f"[cyan]{sid}[/]")
    console.print(tree)


def stop():  # noqa: D401
    global _progress
    if _progress is not None:
        _progress.stop()
        _progress = None 