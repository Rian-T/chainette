from __future__ import annotations
"""Rich live logger – accurate progress bars & DAG tree rendering.

This is the consolidated logging helper (formerly *logging_v3*).  It keeps the
public helpers stable while adding richer progress display. LOC ≈ 110.
"""
from collections import defaultdict
from typing import Dict, Any
from logging import Logger, getLogger, INFO, DEBUG, WARNING, ERROR, basicConfig
from rich.logging import RichHandler

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

__all__ = [
    "show_dag_tree",
    "stop",
    "update_step_badge",
    "get",
    "log",
]

# --------------------------------------------------------------------------- #
# Legacy logger setup (from v1) so existing calls/tests keep working.
# --------------------------------------------------------------------------- #

_LEVEL_MAP = {
    "info": INFO,
    "debug": DEBUG,
    "warning": WARNING,
    "error": ERROR,
}

# Configure root once with Rich handler for plain log messages (non-progress)
basicConfig(
    level=INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

log: Logger = getLogger("chainette")


def get(level: str = "info") -> Logger:  # noqa: D401
    """Return a logger with *level* (str)."""
    lvl = _LEVEL_MAP.get(level.lower(), INFO)
    lg = getLogger("chainette")
    lg.setLevel(lvl)
    return lg

# --------------------------------------------------------------------------- #
# Progress handling
# --------------------------------------------------------------------------- #
_progress: Progress | None = None
_tasks: Dict[str, int] = {}
_counts: Dict[str, int] = defaultdict(int)
_per_item_steps: set[str] = set()


def _ensure_progress() -> Progress:  # noqa: D401
    global _progress
    if _progress is None:
        _progress = Progress(
            TextColumn("[bold blue]{task.fields[id]}[/]"),
            BarColumn(),
            "{task.percentage:>3.0f}%",
            TextColumn("[green]{task.completed}/{task.total}[/]"),
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
    if evt.step_id in _per_item_steps:
        # progress already advanced per item – skip batch advance to avoid double-count
        return
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


# --------------------------------------------------------------------------- #
# Badge helper – optional call from Executor for custom updates
# --------------------------------------------------------------------------- #

def update_step_badge(step_id: str, *, completed: int | None = None, advance: int | None = None):  # noqa: D401
    """Update progress for *step_id*.

    Parameters
    ----------
    completed : int | None
        If given, sets the completed counter to this absolute value.
    advance : int | None
        If given, increments the completed counter by this amount.
    """
    if _progress is None:
        return
    task_id = _tasks.get(step_id)
    if task_id is None:
        return

    if advance is not None:
        _per_item_steps.add(step_id)
        _progress.update(task_id, advance=advance)
    elif completed is not None:
        _progress.update(task_id, completed=completed) 