from __future__ import annotations

"""chainette.cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Minimal Typer CLI exposing warmup, run, kill commands.
"""

import importlib
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from chainette.engine.registry import load_engines_from_yaml
from chainette.engine.runtime import kill_all_engines, kill_engine, spawn_engine

app = typer.Typer(add_completion=False, help="Chainette command‑line interface")


@app.command()
def warmup(
    engines_file: Path = typer.Option(..., "-f", help="YAML file with engine configs"),
    engines: Optional[str] = typer.Option(None, "-e", help="Comma‑separated engine names"),
):
    """Start non‑lazy engines so they are ready for `run`."""

    cfgs = load_engines_from_yaml(engines_file)
    names = set(engines.split(",")) if engines else {c.name for c in cfgs}
    for cfg in cfgs:
        if cfg.name in names and not cfg.lazy:
            live = spawn_engine(cfg)
            print(f"[green]✔[/green] {cfg.name} listening on port {live.port}")


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

    # dynamic import ---------------------------------------------------------
    module_path, obj_name = chain_path.split(":")
    mod = importlib.import_module(module_path)
    chain = getattr(mod, obj_name)

    from datasets import Dataset

    # read inputs ------------------------------------------------------------
    inputs_ds = Dataset.from_json(input_file)
    inputs = [chain.steps[0].input_model.model_validate(row) for row in inputs_ds]

    # load engines -----------------------------------------------------------
    cfgs = load_engines_from_yaml(engines_file) if engines_file else []
    _ = engines  # currently ignored; configs already in registry

    # execute ---------------------------------------------------------------
    ds_dict = chain.run(inputs)

    from chainette.io.writer import RunWriter, flatten_datasetdict

    writer = RunWriter(output_dir, max_lines_per_file=max_lines_per_file, fmt=fmt)
    graph = {}
    for key, ds in ds_dict.items():
        writer.write_step(key, ds)
    flat = flatten_datasetdict(ds_dict)
    writer.write_step("flattened", flat)
    writer.finalize(graph)
    print(f"[green]✔[/green] Run completed under {output_dir}")


@app.command()
def kill(
    engine: Optional[str] = typer.Option(None, "-e", help="Engine name"),
    all_: bool = typer.Option(False, "--all", help="Kill all"),
):
    """Stop one or all engines."""

    if all_:
        kill_all_engines()
        print("[yellow]All engines terminated[/yellow]")
    elif engine:
        kill_engine(engine)
        print(f"[yellow]{engine} terminated[/yellow]")
    else:
        print("[red]Specify -e or --all[/red]")


if __name__ == "__main__":
    app()
