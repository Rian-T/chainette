from __future__ import annotations

"""chainette.cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Minimal Typer CLI exposing warmup, run, kill commands.
"""

import importlib
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from chainette.engine.registry import load_engines_from_yaml
from chainette.engine.runtime import kill_all_engines, kill_engine, spawn_engine
from chainette.utils.constants import SYMBOLS, STYLE

app = typer.Typer(add_completion=False, help="Chainette command‑line interface")
console = Console()

# Import Step at runtime when needed to avoid circular imports
Step = None

@app.command()
def warmup(
    engines_file: Path = typer.Option(..., "-f", help="YAML file with engine configs"),
    engines: Optional[str] = typer.Option(None, "-e", help="Comma‑separated engine names"),
):
    """Start non‑lazy engines so they are ready for `run`."""

    cfgs = load_engines_from_yaml(engines_file)
    names = set(engines.split(",")) if engines else {c.name for c in cfgs}

    console.print(f"\n{SYMBOLS['chain']}Starting engines...", style=STYLE["info"])

    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    table.add_column(style=STYLE["success"])
    table.add_column(style="bold " + STYLE["info"])

    for cfg in cfgs:
        if cfg.name in names and not cfg.lazy:
            live = spawn_engine(cfg)
            table.add_row(SYMBOLS['success'], f"{cfg.name} listening on port {live.port}")

    console.print(table)


@app.command()
def run(
    chain_path: str = typer.Argument(..., help="python_path:obj e.g. examples.qa:my_chain"),
    input_file: Path = typer.Option(..., "-i", exists=True),
    output_dir: Path = typer.Option(..., "--output_dir", file_okay=False),
    max_lines_per_file: int = typer.Option(1000, "--max_lines_per_file"),
    fmt: str = typer.Option("jsonl", "--format", help="jsonl or csv"),
    engines_file: Optional[Path] = typer.Option(None, "-f"),
    engines: Optional[str] = typer.Option(None, "-e"),
):  # noqa: WPS231
    """Load a Chain object and execute it."""

    # Import Step here to avoid circular imports
    global Step
    if Step is None:
        from chainette.core.step import Step

    # dynamic import ---------------------------------------------------------
    module_path, obj_name = chain_path.split(":")
    mod = importlib.import_module(module_path)
    chain = getattr(mod, obj_name)

    from datasets import Dataset

    # read inputs ------------------------------------------------------------
    with console.status("Loading input data..."):
        inputs_ds = Dataset.from_json(input_file)
        first_step = next((s for s in chain.steps if isinstance(s, Step)), None)
        if not first_step:
            console.print(f"{SYMBOLS['error']}Chain does not contain any executable Step.", style=STYLE['error'])
            raise typer.Exit(code=1)
        if not first_step.input_model:
            console.print(f"{SYMBOLS['error']}First Step '{first_step.name}' has no input_model defined.", style=STYLE['error'])
            raise typer.Exit(code=1)

        try:
            inputs = [first_step.input_model.model_validate(row) for row in inputs_ds]
        except Exception as e:
            console.print(f"{SYMBOLS['error']}Failed to validate input data against first step's model ({first_step.input_model.__name__}): {e}", style=STYLE['error'])
            raise typer.Exit(code=1)

    # load engines -----------------------------------------------------------
    cfgs = load_engines_from_yaml(engines_file) if engines_file else []
    _ = engines  # currently ignored; configs already in registry

    # execute ---------------------------------------------------------------
    ds_dict = chain.run(
        inputs,
        output_dir=output_dir,
        max_lines_per_file=max_lines_per_file,
        fmt=fmt
    )

    from chainette.io.writer import flatten_datasetdict

    # Display output summary
    flat = flatten_datasetdict(ds_dict)
    if flat and len(flat) > 0:
        console.print("\n[bold dim]Sample output (first record):[/bold dim]", style=STYLE["dim"])
        sample = flat[0]
        if isinstance(sample, dict):
            sample_table = Table(show_header=True, header_style="bold " + STYLE["info"], box=None, padding=(0, 1))
            sample_table.add_column("Key", style=STYLE["info"])
            sample_table.add_column("Value", style="white")

            for i, (key, value) in enumerate(sample.items()):
                if i >= 5:
                    sample_table.add_row("...", "...")
                    break
                sample_table.add_row(key, str(value)[:80])

            console.print(sample_table)


@app.command()
def kill(
    engine: Optional[str] = typer.Option(None, "-e", help="Engine name"),
    all_: bool = typer.Option(False, "--all", help="Kill all"),
):
    """Stop one or all engines."""

    if all_:
        with console.status("Terminating all engines..."):
            kill_all_engines()
        console.print(f"{SYMBOLS['success']}All engines terminated", style=STYLE["warning"])
    elif engine:
        with console.status(f"Terminating engine '{engine}'..."):
            kill_engine(engine)
        console.print(f"{SYMBOLS['success']}{engine} terminated", style=STYLE["warning"])
    else:
        console.print(f"{SYMBOLS['error']}Specify -e or --all", style=STYLE["error"])


if __name__ == "__main__":
    app()
