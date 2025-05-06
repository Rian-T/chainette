"""
High‑level Chain + Branch execution.

Adds:
* batch_size honoured
* run() forwards output_dir / flattening / file format options
* collects graph metadata (execution order)
"""

from __future__ import annotations

import datetime as _dt
import itertools
import random  # Import random module
from pathlib import Path
from typing import Any, List, Dict, Union, Optional

from datasets import DatasetDict
from pydantic import BaseModel

# Import necessary components from other modules
from chainette.core.step import Step
from chainette.core.apply import ApplyNode
from chainette.core.branch import Branch, Node
from chainette.io.writer import RunWriter, flatten_datasetdict
from chainette.utils.ids import snake_case, new_run_id
from chainette.engine.registry import get_engine_config
from chainette.utils.constants import SYMBOLS, STYLE

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

__all__ = ["Chain"]


class Chain:
    """Represents a sequence of executable steps or branches."""
    def __init__(
        self,
        *,
        name: str,
        steps: List[Node],
        emoji: str = "⛓️",
        batch_size: int = 1,
    ):
        if not name:
            raise ValueError("Chain name cannot be empty.")
        if not steps:
            raise ValueError("Chain must have at least one step.")
        self.name = name
        self.emoji = emoji
        self.steps = steps
        self.batch_size = max(batch_size, 1)
        # Validate input model of the first step
        # Check the first *actual* step, skipping potential initial ApplyNodes
        first_step = next((s for s in steps if isinstance(s, Step)), None)
        if first_step and not first_step.input_model:
             raise ValueError("The first Step node must have an input_model defined.")

    # ------------------------------------------------------------------ #
    # Display Helpers
    # ------------------------------------------------------------------ #

    def _display_banner(self, console: Console):
        """Displays the Chainette ASCII art banner with random colors."""
        # Define a pool of nice colors for the banner - we'll use 4
        colors = [
            "bright_blue", "bright_cyan", "bright_green", "bright_magenta",
            "bright_yellow", "blue", "cyan", "green", "magenta", "yellow",
            "deep_sky_blue1", "spring_green1", "medium_purple1", "orange1"
        ]
        
        # Select exactly 4 random colors from the pool
        selected_colors = random.sample(colors, 4)
        color1, color2, color3, color4 = selected_colors
        
        # Use the 4 colors consistently throughout the banner
        chain_color = color1
        links_color = color2
        border_color = color3
        accent_color = color4  # New fourth color

        # --- ASCII Art Banner ---
        o = f"[{links_color}]o[/]"
        O = f"[{links_color}]O[/]"
        d = f"[{links_color}]~[/]"  # Use ~ for dash
        
        # All letters in CHAINETTE use the same color
        C = f"[{chain_color}]C[/]" 
        H = f"[{chain_color}]H[/]"
        A = f"[{chain_color}]A[/]"
        I = f"[{chain_color}]I[/]"
        N = f"[{chain_color}]N[/]"
        E = f"[{chain_color}]E[/]"
        T = f"[{chain_color}]T[/]"
        
        border = f"[{border_color}]~[/]"
        spacing = f"[{accent_color}] [/]"  # Use the fourth color for spacing between letters

        # Slightly larger banner design
        banner = f"""
    {border*60}
              {o}{d}{d}{O}{d}{d}{o}{d}{d}{O}{d}{d}{o}              
           {o}{d}{d}{O}{d}{d}{o}      {o}{d}{d}{O}{d}{d}{o}           
        {o}{d}{d}{O}{d}{d}{o}            {o}{d}{d}{O}{d}{d}{o}        
          {o}{d}{d}{O}{d}{d}{o}   {C}{spacing}{H}{spacing}{A}{spacing}{I}{spacing}{N}{spacing}{E}{spacing}{T}{spacing}{T}{spacing}{E}   {o}{d}{d}{O}{d}{d}{o}      
        {o}{d}{d}{O}{d}{d}{o}            {o}{d}{d}{O}{d}{d}{o}        
           {o}{d}{d}{O}{d}{d}{o}      {o}{d}{d}{O}{d}{d}{o}           
              {o}{d}{d}{O}{d}{d}{o}{d}{d}{O}{d}{d}{o}              
    {border*60}
    """
        console.print(banner, highlight=False)  # Use highlight=False to prevent Rich from re-interpreting our styles
        # --- End Banner ---

    def _batched(self, items: list[Any]) -> List[List[Any]]:
        """Helper to divide items into batches."""
        if not items:
            return []
        return [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    # ------------------------------------------------------------------ #

    def _calculate_total_steps(self) -> int:
        """Calculate the total number of execution steps for progress tracking."""
        total = 0
        for node in self.steps:
            if isinstance(node, (Step, ApplyNode)):
                # Simple nodes count as 1
                total += 1
            elif isinstance(node, Branch):
                # Count each step in a branch
                total += len(node.steps)
            elif isinstance(node, list):
                # For parallel branches, count the maximum branch length
                # This more accurately reflects time to completion
                max_branch_steps = 0
                for br in node:
                    if isinstance(br, Branch):
                        max_branch_steps = max(max_branch_steps, len(br.steps))
                total += max_branch_steps
            else:
                # Unknown nodes still count as 1 step
                total += 1
        return max(1, total)  # Ensure at least 1 step total

    def _build_execution_plan_tree(self) -> Tree:
        """Create a rich Tree showing the execution plan."""
        console = Console()
        summary_tree = Tree(f"{self.emoji} [bold]{self.name}[/bold] - Execution Plan", style=STYLE["header"])
        steps_tree = summary_tree.add(f"{SYMBOLS['step']}Steps ({len(self.steps)}):")
        
        for i, node in enumerate(self.steps):
            self._add_node_to_tree(steps_tree, node, str(i + 1))
            
        return summary_tree
    
    def _add_node_to_tree(self, parent_tree: Tree, node: Any, index_prefix: str):
        """Helper function to recursively add nodes to the summary tree."""
        node_label = f"Step {index_prefix}"
        node_info = ""
        node_style = ""
        emoji = ""

        if isinstance(node, Step):
            emoji = node.emoji if hasattr(node, 'emoji') else SYMBOLS["step"]
            node_name = node.name or f"Unnamed Step ({node.id})"
            node_info = f"{emoji} {node_label}: [b]{node_name}[/b] ({node.id})"
            node_style = STYLE["node_step"]
            try:
                engine_cfg = get_engine_config(node.engine_name)
                engine_details = f"Engine: [{STYLE['engine']}]{node.engine_name}[/{STYLE['engine']}] (Model: [i]{engine_cfg.model}[/i])"
                parent_tree.add(f"{node_info} [{node_style}] - {engine_details}[/{node_style}]")
            except ValueError:
                parent_tree.add(f"{node_info} [{node_style}] - Engine: [{STYLE['engine']}]{node.engine_name}[/{STYLE['engine']}] (Config not found)[/{node_style}]")

        elif isinstance(node, ApplyNode):
            emoji = SYMBOLS["apply"]
            node_name = node.name or f"Unnamed Apply ({node.id})"
            node_info = f"{emoji} {node_label}: [i]{node_name}[/i] ({node.id})"
            node_style = STYLE["node_apply"]
            parent_tree.add(f"{node_info} [{node_style}]")

        elif isinstance(node, Branch):
            emoji = node.emoji if hasattr(node, 'emoji') else SYMBOLS["branch"]
            node_name = node.name or "Unnamed Branch"
            node_info = f"{emoji} {node_label}: [u]{node_name}[/u]"
            node_style = STYLE["node_branch"]
            branch_subtree = parent_tree.add(f"{node_info} [{node_style}]")
            # Recursively add steps within the branch
            for j, sub_node in enumerate(node.steps):
                self._add_node_to_tree(branch_subtree, sub_node, f"{index_prefix}.{j+1}")

        elif isinstance(node, list):
             # Handle list of parallel nodes (likely branches)
             parallel_subtree = parent_tree.add(f"Step {index_prefix}: Parallel Execution ({len(node)} branches)")
             for j, parallel_node in enumerate(node):
                 # Assume items in the list are executable nodes (like Branch)
                 self._add_node_to_tree(parallel_subtree, parallel_node, f"{index_prefix}.{j+1}")

        else:
             parent_tree.add(f"{SYMBOLS['warning']} {node_label}: Unknown Node Type ({type(node).__name__})")

    def _get_node_description(self, node: Any, i: int, total_steps: int) -> str:
        """Generate a descriptive string for a node."""
        step_label = f"Step {i+1}/{total_steps}"
        
        if isinstance(node, Step):
            return f"{step_label}: '{node.name}'" if node.name else f"{step_label}: ({node.id})"
        elif isinstance(node, ApplyNode):
            return f"{step_label}: '{node.name}'" if node.name else f"{step_label}: ({node.id})"
        elif isinstance(node, Branch):
            return f"{step_label}: '{node.name}'" if node.name else f"{step_label}: Branch"
        elif isinstance(node, list):
            branch_names = [b.name for b in node if isinstance(b, Branch) and hasattr(b, 'name')]
            if branch_names:
                return f"{step_label}: Branching: {', '.join(branch_names)}"
            return f"{step_label}: Branching Point ({len(node)} paths)"
        else:
            return f"{step_label}: (Type: {type(node).__name__})"

    def _execute_node(self, node: Any, current_records: List[Any], writer: RunWriter, 
                     i: int, run_id: str, progress: Progress, chain_task: int, completed_steps: List[int]) -> List[Any]:
        """Execute a single node in the chain and return its outputs."""
        node_info = {}
        
        if isinstance(node, Step):
            node_info = {
                "id": node.id,
                "name": node.name,
                "emoji": getattr(node, 'emoji', SYMBOLS["step"]),
                "type": "step"
            }
            
            if not current_records:
                Console().print(f"{SYMBOLS['warning']}Skipping Step '{node.name}' - empty input", style=STYLE["warning"])
                completed_steps[0] += 1
                progress.update(chain_task, completed=completed_steps[0])
                return []
                
            try:
                outputs = node.run(current_records, run_id=run_id, step_index=i)
                node_info["signature"] = node.signature()
                completed_steps[0] += 1
                progress.update(chain_task, completed=completed_steps[0])
                return outputs
            except Exception as e:
                Console().print(f"{SYMBOLS['error']}Error in Step '{node.name}': {e}", style=STYLE["error"])
                raise
                
        elif isinstance(node, ApplyNode):
            node_info = {
                "id": node.id,
                "name": node.name,
                "emoji": SYMBOLS["apply"],
                "type": "apply"
            }
            
            if not current_records:
                Console().print(f"{SYMBOLS['warning']} Skipping Apply '{node.name}' - empty input", style=STYLE["warning"])
                completed_steps[0] += 1
                progress.update(chain_task, completed=completed_steps[0])
                return []
                
            try:
                outputs = node.run(current_records)
                completed_steps[0] += 1
                progress.update(chain_task, completed=completed_steps[0])
                return outputs
            except Exception as e:
                Console().print(f"{SYMBOLS['error']} Error executing Apply node '{node.name}': {e}", style=STYLE["error"])
                raise
                
        elif isinstance(node, list):
            # This is parallel branch execution
            self._execute_branches(node, current_records, writer, i, run_id, progress, chain_task, completed_steps)
            return []  # Branch execution managed separately
            
        elif isinstance(node, Branch):
            node_info = {
                "id": node.name,
                "name": node.name, 
                "emoji": getattr(node, 'emoji', SYMBOLS["branch"]),
                "type": "branch"
            }
            Console().print(f"{SYMBOLS['warning']} Branch execution logic needs review for '{node.name}'", style=STYLE["warning"])
            completed_steps[0] += 1
            progress.update(chain_task, completed=completed_steps[0])
            return current_records
            
        else:
            Console().print(f"{SYMBOLS['warning']} Skipping unknown node type at index {i}: {type(node)}", style=STYLE["warning"])
            completed_steps[0] += 1
            progress.update(chain_task, completed=completed_steps[0])
            return []
            
        writer.add_node_to_graph({**node_info, "index": i})
        return []

    def _execute_branches(self, branches: List[Branch], current_records: List[Any], 
                         writer: RunWriter, i: int, run_id: str, 
                         progress: Progress, chain_task: int, completed_steps: List[int]) -> None:
        """Execute a list of branches in parallel."""
        console = Console()
        branch_outputs = {}
        num_branches = len(branches)
        
        for br_idx, br in enumerate(branches):
            if not isinstance(br, Branch):
                raise TypeError(f"Lists in Chain.steps must contain Branch objects (found {type(br)})")

            br_records = list(current_records)  # Give each branch a copy of input
            num_sub_steps = len(br.steps)

            for j, sub in enumerate(br.steps):
                sub_step_label = f"Step {i+1}.{br_idx+1}.{j+1}"
                sub_node_name = f"'{sub.name}'" if hasattr(sub, 'name') and sub.name else f"({sub.id})"
                branch_progress_desc = f"Step {i+1}/{len(self.steps)}: Branch [u]{br.name}[/u] ({br_idx+1}/{num_branches}) → {sub_step_label} {sub_node_name}"
                progress.update(chain_task, description=branch_progress_desc)

                if isinstance(sub, Step):
                    try:
                        br_step_outputs = sub.run(br_records, run_id=run_id, step_index=i)
                    except Exception as e:
                        console.print(f"{SYMBOLS['error']}Error in Branch '{br.name}', Step '{sub.name}': {e}", style=STYLE["error"])
                        raise
                    writer.write_step(f"{br.name}.{sub.id}", br_step_outputs)
                    writer.add_node_to_graph({
                        "id": f"{br.name}.{sub.id}", "name": sub.name, "type": "step",
                        "branch": br.name, "signature": sub.signature(),
                    })
                    br_records = br_step_outputs
                    completed_steps[0] += 1
                    progress.update(chain_task, completed=completed_steps[0])

                elif isinstance(sub, ApplyNode):
                    try:
                        br_apply_outputs = sub.run(br_records)
                    except Exception as e:
                        console.print(f"{SYMBOLS['error']}Error in Branch '{br.name}', Apply '{sub.name}': {e}", style=STYLE["error"])
                        raise
                    writer.write_step(f"{br.name}.{sub.id}", br_apply_outputs)
                    writer.add_node_to_graph({
                        "id": f"{br.name}.{sub.id}", "name": sub.name, "type": "apply",
                        "branch": br.name,
                    })
                    br_records = br_apply_outputs
                    completed_steps[0] += 1
                    progress.update(chain_task, completed=completed_steps[0])

                else:
                    raise TypeError(f"Unsupported node {type(sub)} inside branch '{br.name}'")

            branch_outputs[br.name] = br_records
                
        progress.update(chain_task, description=f"Step {i+1}/{len(self.steps)}: Parallel branches completed")

    # ------------------------------------------------------------------ #

    def run(
        self,
        inputs: List[BaseModel],
        *,
        output_dir: str | Path = "output",
        max_lines_per_file: int = 1_000,
        fmt: str = "jsonl",
        generate_flattened_output: bool = True,
    ) -> DatasetDict:
        """Executes the chain on a list of input Pydantic models.

        Args:
            inputs: A list of Pydantic models representing the initial input data.
            output_dir: The root directory where run outputs will be saved.
            max_lines_per_file: Maximum lines per output file (currently affects writer setup).
            fmt: The output file format ('jsonl' or 'csv').
            generate_flattened_output: Whether to generate a single flattened output file.

        Returns:
            A DatasetDict containing the outputs of each step.
        """
        console = Console()
        self._display_banner(console) # Display banner here

        run_id = new_run_id()
        out_root = Path(output_dir) / f"{snake_case(self.name)}_{run_id}"
        writer = RunWriter(out_root, max_lines_per_file, fmt)
        writer.set_chain_name(self.name)

        # --- Pre-Run Summary ---
        summary_tree = self._build_execution_plan_tree()
        summary_tree.add(f"{SYMBOLS['info']}Run ID: [i]{run_id}[/i]")
        summary_tree.add(f"{SYMBOLS['info']}Inputs: {len(inputs)} records")
        summary_tree.add(f"{SYMBOLS['info']}Output Dir: [i]{out_root.resolve()}[/i]")
        summary_tree.add(f"{SYMBOLS['info']}Batch Size: {self.batch_size}")
        
        console.print(Panel(summary_tree, border_style=STYLE["info"], expand=False))
        console.print("") # Spacer
        # --- End Pre-Run Summary ---

        # Write initial inputs
        writer.write_step("input", inputs)
        writer.add_node_to_graph({
            "id": "input",
            "name": "Input Data",
            "type": "input",
            "emoji": "📥",
        })

        current_records: List[Any] = inputs

        # Calculate total execution steps for accurate progress tracking
        total_steps = self._calculate_total_steps()

        # Execute chain steps with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]{task.description}"),
            BarColumn(bar_width=None, complete_style=STYLE["success"]),
            TaskProgressColumn(),  # Shows clearer fraction completed X/Y
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            # Use a mutable object to track progress across function calls
            completed_steps = [0]
            chain_task = progress.add_task("Initializing...", total=total_steps)

            for i, node in enumerate(self.steps):
                desc = self._get_node_description(node, i, len(self.steps))
                progress.update(chain_task, description=desc)

                step_outputs = self._execute_node(node, current_records, writer, 
                                                i, run_id, progress, chain_task, completed_steps)
                
                # Process outputs if any
                if isinstance(node, list):
                    # For branch lists, we've already handled everything in execute_node
                    continue
                    
                if step_outputs:
                    writer.write_step(node.id if hasattr(node, 'id') else node.name, step_outputs)
                    current_records = step_outputs
                else:
                    if isinstance(node, (Step, ApplyNode)):
                        console.print(f"{SYMBOLS['warning']} Node {i} ('{node.name}') produced None/empty output.", style=STYLE["warning"])
                    current_records = []
                    
                if not current_records and i < len(self.steps) - 1:
                    console.print(f"{SYMBOLS['warning']} No records to process. Stopping chain execution.", style=STYLE["warning"])
                    progress.update(chain_task, completed=total_steps, description="Chain stopped early.")
                    break

            # Mark complete at the end
            progress.update(chain_task, description="Chain completed.", completed=total_steps)

        console.print(f"\n{SYMBOLS['chain']}Finalizing output...", style=STYLE["info"])
        ds_dict = writer.finalize(generate_flattened_output=generate_flattened_output)
        console.print(f"{SYMBOLS['success']}Chain '{self.name}' run finished. Output in {out_root.resolve()}", style="bold " + STYLE["success"])
        return ds_dict
