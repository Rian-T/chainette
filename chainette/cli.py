from __future__ import annotations

"""Chainette Command Line Interface."""

import importlib.util
import json
from pathlib import Path
from typing import List, Any, Type

import typer
import logging
from rich.console import Console
from rich.table import Table
from pydantic import BaseModel

from chainette import Chain, Step, ApplyNode, Branch, Node # Assuming __all__ is well-defined
from chainette.engine.registry import _REGISTRY as ENGINE_REGISTRY, EngineConfig, get_engine_config # Accessing internal for simplicity
from chainette.utils import logging as chainette_logging
from chainette.utils.logging import live_logger

app = typer.Typer(
    name="chainette",
    help="CLI for Chainette: typed, lightweight LLM chaining.",
    add_completion=False,
)

# Silence noisy lib logs ASAP
logging.disable(logging.CRITICAL)

console = Console()


def _load_chain_from_file(file_path: Path, chain_name: str) -> Chain:
    """Dynamically load a Chain object from a Python file."""
    if not file_path.exists():
        console.print(f"[bold red]Error: File not found: {file_path}[/]")
        raise typer.Exit(code=1)
    
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        console.print(f"[bold red]Error: Could not load module from {file_path}[/]")
        raise typer.Exit(code=1)
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        console.print(f"[bold red]Error executing Python file {file_path}: {e}[/]")
        raise typer.Exit(code=1)

    if not hasattr(module, chain_name):
        console.print(f"[bold red]Error: Chain '{chain_name}' not found in {file_path}[/]")
        raise typer.Exit(code=1)
    
    chain_obj = getattr(module, chain_name)
    if not isinstance(chain_obj, Chain):
        console.print(f"[bold red]Error: Object '{chain_name}' in {file_path} is not a Chainette Chain.[/]")
        raise typer.Exit(code=1)
    return chain_obj

@app.command()
def engines():
    """List all registered LLM engine configurations."""
    if not ENGINE_REGISTRY:
        console.print("[yellow]No engines registered.[/]")
        return

    table = Table(title="Registered LLM Engines", box=typer.rich_help_panel.box.ROUNDED)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Model Path/ID", style="magenta")
    table.add_column("Dtype", style="green")
    table.add_column("TP Size", style="yellow")
    table.add_column("Max Len", style="blue")
    table.add_column("GPU Util", style="red")
    table.add_column("Reasoning", style="dim")

    for name, config in ENGINE_REGISTRY.items():
        reasoning_status = (
            f"{config.reasoning_parser} (enabled)" 
            if config.enable_reasoning and config.reasoning_parser 
            else "enabled" if config.enable_reasoning 
            else "disabled"
        )
        table.add_row(
            name,
            config.model,
            str(config.dtype or "-"),
            str(config.tensor_parallel_size or "-"),
            str(config.max_model_len or "-"),
            str(config.gpu_memory_utilization or "-"),
            reasoning_status,
        )
    console.print(table)

# Placeholder for inspect and run commands to be added later
@app.command()
def inspect(
    chain_file: Path = typer.Argument(..., help="Path to the Python file containing the chain definition.", exists=True, file_okay=True, dir_okay=False, readable=True),
    chain_name: str = typer.Argument(..., help="Name of the chain variable in the file.")
):
    """Inspect a chain for model compatibility between steps (basic)."""
    console.print(f"[cyan]Inspecting chain '{chain_name}' from {chain_file}...[/]")
    chain_obj = _load_chain_from_file(chain_file, chain_name)
    
    # Simplified inspection logic for now
    # This needs to be significantly more robust for real use
    # For now, we'll just list steps and their declared I/O models
    console.print(f"Chain: [bold]{chain_obj.name}[/] ({len(chain_obj.steps)} top-level nodes)")

    issues_found = 0

    def get_node_io_types(node: Node, prev_output_type: Type[BaseModel] | None) -> tuple[Type[BaseModel] | None, Type[BaseModel] | None, bool]:
        is_compatible = True
        current_input_type = None
        current_output_type = None

        if isinstance(node, Step):
            current_input_type = node.input_model
            current_output_type = node.output_model
            if prev_output_type and prev_output_type != current_input_type:
                is_compatible = False
        elif isinstance(node, ApplyNode):
            # Check if we have type information in the ApplyNode
            if hasattr(node, 'input_model') and node.input_model:
                current_input_type = node.input_model
                # Compare with previous step's output
                if prev_output_type and prev_output_type != current_input_type:
                    is_compatible = False
            else:
                # If no input_model specified, assume it can accept previous output
                current_input_type = prev_output_type

            if hasattr(node, 'output_model') and node.output_model:
                current_output_type = node.output_model
            else:
                # Without explicit output_model, we can't determine
                current_output_type = Any
                console.print(f"  [yellow]Warning:[/] ApplyNode '{node.name}' output model type not specified. Assuming compatible.")
        elif isinstance(node, Branch):
            # For a branch, its steps run on the same input as the branch received.
            # The output of the branch is the output of its last step.
            branch_prev_output = prev_output_type
            for sub_node_idx, sub_node in enumerate(node.steps):
                _, branch_prev_output, _ = get_node_io_types(sub_node, branch_prev_output)
            current_input_type = prev_output_type # Branch takes input from previous step
            current_output_type = branch_prev_output
            # Compatibility check for branch itself relies on its internal steps

        return current_input_type, current_output_type, is_compatible

    previous_node_output_type: Type[BaseModel] | None = None # Input to the very first step

    for i, node_or_branch_list in enumerate(chain_obj.steps):
        if isinstance(node_or_branch_list, list): # It's a list of branches
            console.print(f"Node {i+1}: Parallel Branches ({len(node_or_branch_list)})")
            for branch_idx, branch_node in enumerate(node_or_branch_list):
                console.print(f"  Branch {branch_idx + 1}: '{branch_node.name}'")
                branch_input_type = previous_node_output_type # Each branch gets same input
                branch_output_type_final = branch_input_type
                for sub_i, sub_node in enumerate(branch_node.steps):
                    sub_node_input_type, sub_node_output_type, compatible = get_node_io_types(sub_node, branch_output_type_final)
                    type_info = f"{sub_node_input_type.__name__ if sub_node_input_type else 'Dynamic'} -> {sub_node_output_type.__name__ if sub_node_output_type else 'Dynamic'}"
                    status = "[green]OK[/]" if compatible else "[bold red]MISMATCH[/]"
                    if not compatible: issues_found += 1
                    console.print(f"    Step {sub_i+1} ('{sub_node.name}' - {type(sub_node).__name__}): {type_info} - {status}")
                    branch_output_type_final = sub_node_output_type
            # Output of parallel branches is not fed back to main chain in this design.
        elif isinstance(node_or_branch_list, Node):
            node = node_or_branch_list
            node_input_type, node_output_type, compatible = get_node_io_types(node, previous_node_output_type)
            
            type_info = f"{node_input_type.__name__ if node_input_type else 'Dynamic'} -> {node_output_type.__name__ if node_output_type else 'Dynamic'}"
            status = "[green]OK[/]" if compatible else "[bold red]MISMATCH[/]"
            if not compatible: issues_found +=1

            console.print(f"Node {i+1} ('{node.name}' - {type(node).__name__}): {type_info} - {status}")
            previous_node_output_type = node_output_type
        else:
            console.print(f"[red]Unknown node type at step {i+1}[/]")
            issues_found += 1

    if issues_found == 0:
        console.print("[bold green]Chain inspection passed. All declared Step I/O models are compatible.[/]")
    else:
        console.print(f"[bold red]Chain inspection failed. Found {issues_found} potential I/O model mismatch(es).[/]")
        raise typer.Exit(code=1)

@app.command()
def run(
    chain_file: Path = typer.Argument(..., help="Path to the Python file containing the chain definition.", exists=True, file_okay=True, dir_okay=False, readable=True),
    chain_name: str = typer.Argument(..., help="Name of the chain variable in the file."),
    input_file: Path = typer.Argument(None, help="Path to the input JSONL file (each line is a Pydantic model for the first step).", exists=False),
    hf_dataset: str = typer.Option(None, "--hf-dataset", help="HuggingFace dataset path (repo_id or local folder)"),
    hf_split: str = typer.Option("train", "--hf-split", help="Split name or slicing notation (e.g. train[:1000])"),
    output_dir: Path = typer.Argument(..., help="Directory to save the output datasets.", file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    generate_flattened: bool = typer.Option(True, "--flattened/--no-flattened", help="Generate a single flattened output file."),
    max_lines_per_file: int = typer.Option(1000, help="Maximum lines per output data file."),
    no_icons: bool = typer.Option(False, "--no-icons", help="Disable emoji/icons in DAG tree."),
    max_branches: int = typer.Option(None, "--max-branches", help="Limit number of branches shown under parallel wrapper."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON event logs instead of Rich UI."),
):
    """Run a chain with inputs from a JSONL file and save results."""
    console.print(f"[cyan]Running chain '{chain_name}' from {chain_file}...[/]")
    chain_obj = _load_chain_from_file(chain_file, chain_name)

    # Determine the input model type for the first step of the chain
    first_node = None
    if chain_obj.steps:
        if isinstance(chain_obj.steps[0], list): # Parallel branches first
            if chain_obj.steps[0] and chain_obj.steps[0][0].steps:
                 first_node = chain_obj.steps[0][0].steps[0]
        elif isinstance(chain_obj.steps[0], Node):
            first_node = chain_obj.steps[0]
    
    input_model_type: Type[BaseModel] | None = None
    if isinstance(first_node, Step):
        input_model_type = first_node.input_model
    elif isinstance(first_node, ApplyNode):
        if getattr(first_node, "input_model", None):
            input_model_type = first_node.input_model  # type: ignore[assignment]
            console.print("[yellow]First step is ApplyNode – using declared input_model for parsing.[/]")
        else:
            console.print("[yellow]Warning: First step is an ApplyNode without input_model. Unable to infer type.[/]")
    elif isinstance(first_node, Branch):
         if first_node.steps and isinstance(first_node.steps[0], Step):
             input_model_type = first_node.steps[0].input_model

    raw_mode = input_model_type is None
    if raw_mode:
        console.print("[yellow]Input model could not be inferred – loading inputs as raw JSON dicts.[/]")

    # Load inputs
    inputs_data: List[BaseModel] = []
    if hf_dataset:
        try:
            from datasets import load_dataset  # lazy import
            ds = load_dataset(hf_dataset, split=hf_split)
            console.print(f"Loaded HF dataset '{hf_dataset}' split '{hf_split}' (rows={len(ds)}).")
            for row in ds:
                data_obj = row
                if not raw_mode:
                    try:
                        data_obj = input_model_type.model_validate(row)
                    except Exception:  # fallback keep raw
                        pass
                inputs_data.append(data_obj)
        except Exception as e:
            console.print(f"[bold red]Failed to load HuggingFace dataset: {e}[/]")
            raise typer.Exit(code=1)
    else:
        # JSONL path
        if input_file is None or not input_file.exists():
            console.print("[bold red]Input file not found.[/]")
            raise typer.Exit(code=1)
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        data = json.loads(line)
                        if raw_mode:
                            inputs_data.append(data)
                        else:
                            try:
                                inputs_data.append(input_model_type.model_validate(data))
                            except Exception as e:  # noqa: BLE001
                                console.print(f"[bold red]Error validating line {line_num}: {e}[/]")
                                raise typer.Exit(code=1)
            console.print(f"Loaded {len(inputs_data)} input records from {input_file}.")
        except Exception as e:
            console.print(f"[bold red]Error reading input file {input_file}: {e}[/]")
            raise typer.Exit(code=1)

    if not inputs_data:
        console.print("[yellow]Warning: Input file is empty or contains no valid data. Nothing to run.[/]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Always show banner and DAG before execution (quiet mode removed)
    from chainette.utils.banner import ChainetteBanner
    ChainetteBanner(console=console).display()
    console.print(
        f"[bold]Chain:[/] {chain_obj.name} • [bold]Inputs:[/] {len(inputs_data)} • "
        f"[bold]Output Dir:[/] {output_dir}",
    )
    console.print("")
    chainette_logging.show_dag_tree(chain_obj, no_icons=no_icons, max_branches=max_branches)

    try:
        if not json_logs:
            live_logger.start()

        # StreamWriter is now the default and only writer mode.
        from chainette.io.stream_writer import StreamWriter
        writer = StreamWriter(output_dir, max_lines_per_file=max_lines_per_file, fmt="jsonl")

        chain_obj.run(
            inputs_data,
            writer=writer,
            generate_flattened_output=generate_flattened,
        )
    finally:
        if not json_logs:
            live_logger.stop()

    console.print(f"\n[bold green]Chain execution finished successfully![/]")
    console.print(f"Results written to {output_dir}")

# --------------------------------------------------------------------------- #
# YAML runner
# --------------------------------------------------------------------------- #

from chainette.yaml_loader import load_chain as _load_chain_yaml


@app.command("run-yaml")
def run_yaml(
    yaml_file: Path = typer.Argument(..., help="YAML chain file", exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Argument(..., help="Output directory", dir_okay=True, file_okay=False, writable=True, resolve_path=True),
    generate_flattened: bool = typer.Option(True, "--flattened/--no-flattened"),
    max_lines_per_file: int = typer.Option(1000),
    symbols_module: str = typer.Option("examples.ollama_gemma_features", help="Python module with step/branch symbols"),
    hf_dataset: str = typer.Option(None, "--hf-dataset", help="HuggingFace dataset path"),
    hf_split: str = typer.Option("train", "--hf-split"),
):
    """Run a chain defined in YAML."""
    console.print(f"[cyan]Running YAML chain from {yaml_file}…[/]")

    syms_mod = importlib.import_module(symbols_module)
    chain_obj = _load_chain_yaml(yaml_file, symbols=syms_mod.__dict__)

    # Collect first step's input model for hint
    first = chain_obj.steps[0]
    console.print(f"Chain '[bold]{chain_obj.name}[/]' loaded with {len(chain_obj.steps)} top-level steps.")

    # Load inputs (HF dataset or default inputs2.jsonl)
    if hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(hf_dataset, split=hf_split)
        console.print(f"Loaded HF dataset '{hf_dataset}' split '{hf_split}' (rows={len(ds)}).")
        inputs_raw = list(ds)
    else:
        input_path = Path("inputs2.jsonl")
        if not input_path.exists():
            console.print("[red]inputs2.jsonl not found – provide --hf-dataset or create file.[/]")
            raise typer.Exit(1)
        import json
        inputs_raw = [json.loads(l) for l in input_path.read_text().splitlines() if l]

    model_cls = first.input_model if isinstance(first, Node) else None  # type: ignore[attr-defined]
    if not model_cls:
        console.print("[red]Cannot infer input model.[/]")
        raise typer.Exit(1)

    pyd_inputs = []
    for obj in inputs_raw:
        try:
            pyd_inputs.append(model_cls.model_validate(obj))
        except Exception:
            pyd_inputs.append(obj)

    chain_obj.run(pyd_inputs, output_dir=output_dir, generate_flattened_output=generate_flattened, max_lines_per_file=max_lines_per_file)

    console.print("[green]YAML chain finished.[/]")

# --------------------------------------------------------------------------- #
# New DAG-only inspector
# --------------------------------------------------------------------------- #


@app.command("inspect-dag")
def inspect_dag(
    chain_file: Path = typer.Argument(..., help="Python file containing chain."),
    chain_name: str = typer.Argument(..., help="Chain variable name."),
):
    """Print a Rich DAG tree without executing the chain."""
    chain_obj = _load_chain_from_file(chain_file, chain_name)

    from chainette.utils.banner import ChainetteBanner
    ChainetteBanner(console=console).display()

    console.print(f"[bold]Chain:[/] {chain_obj.name} • [bold]DAG ONLY[/]")
    console.print("")

    from chainette.utils.logging import show_dag_tree  # noqa: WPS433
    from chainette.utils.dag import RenderOptions

    opts = RenderOptions(icons_on=True, max_branches=max_branches)
    show_dag_tree(chain_obj, opts=opts)
    console.print(f"[green]DAG inspection complete – {len(chain_obj.steps)} top-level nodes shown.[/]")

# --------------------------------------------------------------------------- #
# Warm-up non-lazy engines for a Chain                                        #
# --------------------------------------------------------------------------- #


@app.command("warmup")
def warmup(
    chain_file: Path = typer.Argument(..., help="Python or YAML file defining the chain."),
    chain_name: str = typer.Option(None, "--chain-name", help="Name of the Chain variable when using a .py file."),
):
    """Instantiate all *non-lazy* engines used by the specified chain.

    • For **.py** files you must pass `--chain-name` pointing to the Chain object.
    • For **.yml/.yaml** definitions we call `yaml_loader.load_chain`.

    The command simply triggers the lazy `cfg.engine` property so HTTP clients
    get instantiated and (in future) any required vLLM server processes are
    spawned.  Currently it only instantiates clients – process management will
    be added once `EngineProcessManager` lands.
    """

    console.print("[cyan]Warming up engines…[/]")

    # Resolve chain object
    if chain_file.suffix in {".yml", ".yaml"}:
        from chainette.yaml_loader import load_chain as _load_chain_yaml  # noqa: WPS433

        chain_obj = _load_chain_yaml(chain_file)
    else:
        if chain_name is None:
            console.print("[red]--chain-name is required for Python files.[/]")
            raise typer.Exit(1)
        chain_obj = _load_chain_from_file(chain_file, chain_name)

    # Collect unique engine configs
    from chainette.core.step import Step  # lazy import

    engine_names: set[str] = set()
    for node in chain_obj.steps:
        nodes = node if not isinstance(node, list) else node  # flatten top-level
        for sub in (nodes if isinstance(nodes, list) else [nodes]):
            if isinstance(sub, Step):
                engine_names.add(sub.engine_name)

    warmed = 0
    from chainette.engine.registry import get_engine_config

    for name in engine_names:
        cfg = get_engine_config(name)
        if cfg.lazy:
            continue  # skip lazy engines
        _ = cfg.engine  # instantiate
        warmed += 1
        console.print(f"[green]✓[/] {name} ({cfg.backend}) ready")

    if warmed == 0:
        console.print("[yellow]No non-lazy engines found – nothing to warm up.[/]")
    else:
        console.print(f"[bold green]Warm-up complete – {warmed} engine(s) ready.[/]")

if __name__ == "__main__":
    app() 