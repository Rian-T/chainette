from __future__ import annotations
"""Rich live logger – accurate progress bars & DAG tree rendering.

This implementation uses a rich.live.Live display with a Layout to show
a logging panel and a progress panel simultaneously and without conflict.
"""
from collections import defaultdict
from typing import Dict, Any, List
from logging import Logger, getLogger, INFO, DEBUG, WARNING, ERROR, basicConfig
from rich.logging import RichHandler

from rich.console import Console
from rich.tree import Tree
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from chainette.utils.events import (
    subscribe,
    BatchFinished,
    StepTotalItems,
    EngineStarted,
    EngineReleased,
    EngineLogReceived,
)
from chainette.utils.dag import build_rich_tree

console = Console()

__all__ = [
    "show_dag_tree",
    "stop",
    "update_step_badge",
    "get",
    "log",
    "live_logger",
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

class LiveLogger:
    """Manages a rich.live.Live display with a Layout for logs and progress."""
    def __init__(self):
        self.layout = self._make_layout()
        self.live = Live(self.layout, console=console, transient=True, auto_refresh=False)
        self.log_messages: List[Text] = []
        self.progress: Progress | None = None
        self._tasks: Dict[str, int] = {}
        self._per_item_steps: set[str] = set()

    def _make_layout(self) -> Layout:
        """Creates the initial layout for the live display."""
        layout = Layout()
        layout.split_column(
            Layout(name="logs", size=5),
            Layout(name="progress"),
        )
        layout["logs"].update(Panel(Text(""), title="[bold]Logs[/]", border_style="dim"))
        layout["progress"].update(Panel(Text(""), title="[bold]Progress[/]", border_style="dim"))
        return layout

    def start(self):
        """Starts the live display."""
        self.live.start()

    def stop(self):
        """Stops the live display and prints a final summary log."""
        if self.live.is_started:
            self.live.stop()

        # Flush engines first to capture all release events before processing logs
        from chainette.engine.broker import EngineBroker
        EngineBroker.flush(force=True)

        # --- Process all collected log messages ---
        started_engines = []
        released_engines = []
        process_logs = []

        for msg_text in self.log_messages:
            msg_str = msg_text.plain
            if msg_str.startswith("🚀 Started engine"):
                try:
                    name = msg_str.split("'")[1]
                    started_engines.append(name)
                except IndexError:
                    pass  # Should not happen
            elif msg_str.startswith("🗑️ Released engine"):
                try:
                    name = msg_str.split("'")[1]
                    released_engines.append(name)
                except IndexError:
                    pass
            else:
                # It's a process log from a server
                process_logs.append(msg_str)

        # --- Print the cool ending script ---
        console.rule("[bold cyan]Execution Summary[/]", style="cyan")

        if started_engines:
            console.print(f"🚀 [bold green]Engines Started:[/] {', '.join(sorted(list(set(started_engines))))}")

        # Filter for important process logs
        important_logs = [log for log in process_logs if "error" in log.lower() or "warn" in log.lower()]
        if important_logs:
            console.print("\n[bold yellow]Notable Process Logs:[/]")
            for log in important_logs:
                console.print(f"  [dim]{log}[/dim]")

        if released_engines:
            console.print(f"🗑️ [bold red]Engines Released:[/] {', '.join(sorted(list(set(released_engines))))}")
            
        self.log_messages.clear()

    def _update_log_panel(self):
        """Updates the log panel with the latest messages."""
        log_renderable = Text("\n").join(self.log_messages[-4:]) # Show last 4 messages
        self.layout["logs"].update(
            Panel(log_renderable, title="[bold]Logs[/]", border_style="dim")
        )
        self.live.refresh()

    def _ensure_progress(self):
        if self.progress is None:
            self.progress = Progress(
                TextColumn("[bold blue]{task.fields[id]}[/]"),
                BarColumn(),
                "{task.percentage:>3.0f}%",
                TextColumn("[green]{task.completed}/{task.total}[/]"),
                "•",
                TimeElapsedColumn(),
            )
            self.layout["progress"].update(
                Panel(self.progress, title="[bold]Progress[/]", border_style="dim")
            )
        return self.progress
        
    def on_total(self, evt: StepTotalItems):
        prog = self._ensure_progress()
        if evt.step_id not in self._tasks:
            self._tasks[evt.step_id] = prog.add_task(
                description="", total=evt.total, id=evt.step_id
            )
        else:
            prog.update(self._tasks[evt.step_id], total=evt.total)
        self.live.refresh()
        
    def on_batch_finish(self, evt: BatchFinished):
        if self.progress and evt.step_id in self._tasks:
            self.progress.update(self._tasks[evt.step_id], advance=evt.count)
            self.live.refresh()

    def on_engine_started(self, ev: EngineStarted):
        self.log_messages.append(
            Text(f"🚀 Started engine '{ev.engine_name}' ({ev.backend})", style="green")
        )
        self._update_log_panel()

    def on_engine_released(self, ev: EngineReleased):
        self.log_messages.append(
            Text(f"🗑️ Released engine '{ev.engine_name}' ({ev.backend})", style="red")
        )
        # Don't refresh here, as this happens during teardown

    def on_engine_log(self, ev: EngineLogReceived):
        """Append an engine's log message to the log panel."""
        self.log_messages.append(
            Text(f"[{ev.engine_name}] {ev.message}", style="dim")
        )
        self._update_log_panel()

# Global instance
live_logger = LiveLogger()

# Wire up subscribers to the instance methods
subscribe(StepTotalItems)(live_logger.on_total)
subscribe(BatchFinished)(live_logger.on_batch_finish)
subscribe(EngineStarted)(live_logger.on_engine_started)
subscribe(EngineReleased)(live_logger.on_engine_released)
subscribe(EngineLogReceived)(live_logger.on_engine_log)

# Public-facing functions that delegate to the global instance
def show_dag_tree(obj: Any, **kw):
    # This should be printed before the live display starts
    console.print(build_rich_tree(obj, **kw))

def stop():
    live_logger.stop()

def update_step_badge(step_id: str, *, completed: int | None = None, advance: int | None = None):
    if live_logger.progress and step_id in live_logger._tasks:
        task_id = live_logger._tasks[step_id]
        if advance is not None:
            live_logger._per_item_steps.add(step_id)
            live_logger.progress.update(task_id, advance=advance)
        elif completed is not None:
            live_logger.progress.update(task_id, completed=completed)
        live_logger.live.refresh() 