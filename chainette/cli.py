from __future__ import annotations

"""chainette.cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Minimal Typer CLI exposing warmup, run, kill commands.
"""

import importlib
import json
from pathlib import Path
from typing import Optional, List, Type, Union, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from pydantic import BaseModel

from chainette.engine.registry import load_engines_from_yaml
from chainette.engine.runtime import kill_all_engines, kill_engine, spawn_engine
from chainette.utils.constants import SYMBOLS, STYLE

app = typer.Typer(add_completion=False, help="Chainette command‑line interface")
console = Console()

# Import Step at runtime when needed to avoid circular imports
Step = None
Chain = None
ApplyNode = None
Branch = None

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


@app.command()
def inspect(
    chain_path: str = typer.Argument(..., help="python_path:obj e.g. examples.qa:my_chain"),
):
    """Inspect and validate a Chain without running it."""
    global Step, Chain, ApplyNode, Branch
    if Step is None: from chainette.core.step import Step
    if Chain is None: from chainette.core.chain import Chain
    if ApplyNode is None: from chainette.core.apply import ApplyNode
    if Branch is None: from chainette.core.branch import Branch

    try:
        module_path, obj_name = chain_path.split(":")
        mod = importlib.import_module(module_path)
        chain_obj = getattr(mod, obj_name)
    except (ValueError, ImportError, AttributeError) as e:
        console.print(f"{SYMBOLS['error']}Failed to import chain from '{chain_path}': {e}", style=STYLE["error"])
        raise typer.Exit(code=1)

    if not isinstance(chain_obj, Chain):
        console.print(f"{SYMBOLS['error']}Object '{obj_name}' is not a Chain instance", style=STYLE["error"])
        raise typer.Exit(code=1)

    console.print(f"\n{SYMBOLS['chain']}Inspecting chain: [bold]{chain_obj.name}[/bold]", style=STYLE["info"])
    summary_tree = chain_obj._build_execution_plan_tree()
    console.print(Panel(summary_tree, border_style=STYLE["info"], expand=False))
    
    validation_issues = []

    _perform_validation_for_sequence(
        sequence_name=f"Main Chain: {chain_obj.name}",
        nodes=chain_obj.steps,
        input_to_sequence=None, # Will be derived from the first step of the chain
        base_path_for_issues="", # Full path for issue reporting
        console_obj=console,
        validation_issues=validation_issues
    )

    if validation_issues:
        console.print(f"\n{SYMBOLS['error']}Found {len(validation_issues)} validation issues across all contexts:", style=STYLE["error"])
        for issue in validation_issues:
            console.print(f"  • {issue}", style=STYLE["error"])
    else:
        console.print(f"\n{SYMBOLS['success']}Chain validation successful! All model connections appear compatible.", style=STYLE["success"])


def _get_node_name_for_path(node_instance: Any, index: int) -> str:
    """Helper to get a descriptive name for a node for path construction."""
    if isinstance(node_instance, list):
        if not node_instance: return "EmptyParallelBlock"
        is_parallel_steps = all(isinstance(n, (Step, ApplyNode)) for n in node_instance)
        return "ParallelSteps" if is_parallel_steps else "ParallelBranches"
    return getattr(node_instance, 'name', None) or \
           (getattr(node_instance, 'id', None) if isinstance(node_instance, (Step, ApplyNode)) else None) or \
           f"Unnamed_{type(node_instance).__name__}"


def _perform_validation_for_sequence(
    sequence_name: str,
    nodes: List[Any],
    input_to_sequence: Optional[Type[BaseModel]],
    base_path_for_issues: str, # Used for constructing full paths for the validation_issues list
    console_obj: Console,
    validation_issues: List[str]
):
    console_obj.print(f"\n{SYMBOLS['info']}Validating Sequence: [bold]{sequence_name}[/bold]", style=STYLE["info"])
    
    table = Table(show_header=True, header_style="bold " + STYLE["info"], title=sequence_name)
    table.add_column("Node Path (Relative)", overflow="fold")
    table.add_column("Input Model")
    table.add_column("Output Model")
    table.add_column("Status")
    table.add_column("Details", overflow="fold")

    class _DictOutputType(BaseModel): pass
    _DictOutputType.__name__ = "Dict[str,Any]"

    current_sequence_effective_input = input_to_sequence
    if not current_sequence_effective_input and nodes: # Try to derive for the very first sequence (main chain)
        first_node_in_chain = nodes[0]
        if isinstance(first_node_in_chain, Step):
            current_sequence_effective_input = first_node_in_chain.input_model
        elif isinstance(first_node_in_chain, list) and first_node_in_chain: # Parallel block starts
            first_item_in_parallel = first_node_in_chain[0]
            if isinstance(first_item_in_parallel, Step):
                current_sequence_effective_input = first_item_in_parallel.input_model
            elif isinstance(first_item_in_parallel, Branch) and first_item_in_parallel.steps and isinstance(first_item_in_parallel.steps[0], Step):
                 current_sequence_effective_input = first_item_in_parallel.steps[0].input_model
        elif isinstance(first_node_in_chain, Branch) and first_node_in_chain.steps and isinstance(first_node_in_chain.steps[0], Step):
            current_sequence_effective_input = first_node_in_chain.steps[0].input_model


    table.add_row(
        "SequenceInput",
        "-",
        current_sequence_effective_input.__name__ if current_sequence_effective_input else "Any/Untyped",
        f"[{STYLE['info']}]INFO[/{STYLE['info']}]",
        f"Input to this sequence."
    )

    effective_prev_output_model = current_sequence_effective_input

    for i, node_instance in enumerate(nodes):
        node_name_str = _get_node_name_for_path(node_instance, i)
        relative_node_path = f"[{i}]::{node_name_str}"
        full_node_path_for_issue = f"{base_path_for_issues}{relative_node_path}"

        if isinstance(node_instance, Step):
            step_input_model = node_instance.input_model
            step_output_model = node_instance.output_model
            input_model_name = step_input_model.__name__ if step_input_model else "N/A"
            output_model_name = step_output_model.__name__ if step_output_model else "N/A"
            status_style, status_text, details = STYLE['info'], "INFO", ""

            if not step_input_model:
                status_style, status_text, details = STYLE['error'], "ERROR", "Step has no input_model defined."
                validation_issues.append(f"{full_node_path_for_issue}: {details}")
            elif effective_prev_output_model:
                if step_input_model == effective_prev_output_model:
                    status_style, status_text, details = STYLE['success'], "OK", f"Matches previous output ({effective_prev_output_model.__name__})."
                elif effective_prev_output_model == _DictOutputType and step_input_model != dict:
                    status_style, status_text, details = STYLE['error'], "ERROR", f"Input '{input_model_name}' received Dict[str,Any] from parallel block. Ensure compatibility or use an adapter."
                    validation_issues.append(f"{full_node_path_for_issue}: {details}")
                else:
                    status_style, status_text, details = STYLE['error'], "ERROR", f"Input '{input_model_name}' mismatches previous output '{effective_prev_output_model.__name__}'."
                    validation_issues.append(f"{full_node_path_for_issue}: {details}")
            else: # No previous output model to compare against (e.g. first step after untyped ApplyNode)
                details = "First step in sequence or follows untyped node."
            
            if not step_output_model:
                current_status_is_error = status_text == "ERROR"
                status_style, status_text = STYLE['error'], "ERROR"
                details = (details + " | " if details and not current_status_is_error else "") + "Step has no output_model defined."
                validation_issues.append(f"{full_node_path_for_issue}: No output_model defined.")

            table.add_row(relative_node_path, input_model_name, output_model_name, f"[{status_style}]{status_text}[/{status_style}]", details)
            effective_prev_output_model = step_output_model if status_text != "ERROR" else None

        elif isinstance(node_instance, ApplyNode):
            table.add_row(relative_node_path, "Any", "Any", f"[{STYLE['info']}]INFO[/{STYLE['info']}]", "ApplyNode: Untyped I/O. Output is considered Any.")
            effective_prev_output_model = None

        elif isinstance(node_instance, Branch):
            table.add_row(relative_node_path, effective_prev_output_model.__name__ if effective_prev_output_model else "Any", "↪ (see sub-table)", f"[{STYLE['info']}]BRANCH[/{STYLE['info']}]", f"Details for branch '{node_name_str}' in separate table.")
            
            branch_output_model = _perform_validation_for_sequence(
                sequence_name=f"Branch: {node_name_str} (from {full_node_path_for_issue})",
                nodes=node_instance.steps,
                input_to_sequence=effective_prev_output_model,
                base_path_for_issues=f"{full_node_path_for_issue}.",
                console_obj=console_obj,
                validation_issues=validation_issues
            )
            effective_prev_output_model = branch_output_model

        elif isinstance(node_instance, list): # Parallel Block
            block_input_name = effective_prev_output_model.__name__ if effective_prev_output_model else "Any"
            table.add_row(relative_node_path, block_input_name, _DictOutputType.__name__, f"[{STYLE['info']}]PARALLEL[/{STYLE['info']}]", f"{node_name_str} block. Output is Dict[str,Any]. Items validated below if Steps, or in sub-tables if Branches.")

            is_parallel_steps = node_name_str == "ParallelSteps"

            if is_parallel_steps:
                for sub_i, sub_node in enumerate(node_instance): # sub_node is Step or ApplyNode
                    sub_node_name = _get_node_name_for_path(sub_node, sub_i)
                    sub_relative_path = f"{relative_node_path}.[{sub_i}]::{sub_node_name}"
                    sub_full_path_issue = f"{full_node_path_for_issue}.[{sub_i}]::{sub_node_name}"

                    if isinstance(sub_node, Step):
                        s_input, s_output = sub_node.input_model, sub_node.output_model
                        s_in_name, s_out_name = (s_input.__name__ if s_input else "N/A"), (s_output.__name__ if s_output else "N/A")
                        s_style, s_text, s_details = STYLE['info'], "INFO", ""
                        if not s_input:
                            s_style, s_text, s_details = STYLE['error'], "ERROR", "No input_model."
                            validation_issues.append(f"{sub_full_path_issue}: {s_details}")
                        elif effective_prev_output_model and s_input != effective_prev_output_model:
                            s_style, s_text, s_details = STYLE['error'], "ERROR", f"Input '{s_in_name}' mismatches block input '{block_input_name}'."
                            validation_issues.append(f"{sub_full_path_issue}: {s_details}")
                        else:
                            s_style, s_text, s_details = STYLE['success'], "OK", f"Input matches block input ({block_input_name})."
                        if not s_output: # Should be caught by Step init
                             s_style, s_text = STYLE['error'], "ERROR"; s_details += " | No output_model."
                             validation_issues.append(f"{sub_full_path_issue}: No output_model.")
                        table.add_row(sub_relative_path, s_in_name, s_out_name, f"[{s_style}]{s_text}[/{s_style}]", s_details)
                    elif isinstance(sub_node, ApplyNode):
                        table.add_row(sub_relative_path, "Any", "Any", f"[{STYLE['info']}]INFO[/{STYLE['info']}]", "Parallel ApplyNode.")
            
            elif node_name_str == "ParallelBranches": # List of Branches
                for sub_i, sub_branch_node in enumerate(node_instance):
                    if isinstance(sub_branch_node, Branch):
                        sub_branch_name = _get_node_name_for_path(sub_branch_node, sub_i)
                        sub_branch_relative_path_marker = f"{relative_node_path}.[{sub_i}]::{sub_branch_name}"
                        sub_branch_full_path_issue_base = f"{full_node_path_for_issue}.[{sub_i}]::{sub_branch_name}"
                        
                        table.add_row(sub_branch_relative_path_marker, block_input_name, "↪ (see sub-table)", f"[{STYLE['info']}]BRANCH[/{STYLE['info']}]", f"Parallel branch '{sub_branch_name}'. Details in sub-table.")
                        
                        _perform_validation_for_sequence(
                            sequence_name=f"Parallel Branch: {sub_branch_name} (from {sub_branch_full_path_issue_base})",
                            nodes=sub_branch_node.steps,
                            input_to_sequence=effective_prev_output_model, # Input to the parallel block
                            base_path_for_issues=f"{sub_branch_full_path_issue_base}.",
                            console_obj=console_obj,
                            validation_issues=validation_issues
                        )
            effective_prev_output_model = _DictOutputType
        else: # Unknown node type
            table.add_row(relative_node_path, "?", "?", f"[{STYLE['warning']}]UNKNOWN[/{STYLE['warning']}]", f"Unknown node type: {type(node_instance).__name__}")
            validation_issues.append(f"{full_node_path_for_issue}: Unknown node type {type(node_instance).__name__}")
            effective_prev_output_model = None
            
    console_obj.print(table)
    return effective_prev_output_model # Return the output model of the last processed node in this sequence

if __name__ == "__main__":
    app()
