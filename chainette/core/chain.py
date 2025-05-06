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
from chainette.utils.banner import ChainetteBanner  # Import the new banner class

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
        """Displays the Chainette ASCII art banner."""
        # Use a specific set of colors for the banner for consistency,
        # or allow ChainetteBanner to use its defaults.
        # For this example, we'll use some predefined "nice" colors.
        banner = ChainetteBanner(
            console=console,
            # Example of customizing colors if desired:
            # border_color="deep_sky_blue1",
            # links_color="medium_purple1",
            # accent_color="orange1"
        )
        banner.display()

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
                if not node:  # Empty list
                    total += 1
                    continue
                # Check if it's a list of parallel steps/applynodes or parallel branches
                is_parallel_steps = all(isinstance(n, (Step, ApplyNode)) for n in node)
                if is_parallel_steps:
                    # Parallel steps/applynodes count as 1 logical step
                    total += 1
                else:  # Assuming parallel branches
                    # For parallel branches, count the maximum branch length
                    max_branch_steps = 0
                    for br in node:
                        if isinstance(br, Branch):
                            max_branch_steps = max(max_branch_steps, len(br.steps))
                    total += max_branch_steps if max_branch_steps > 0 else 1
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
            if not node:
                parent_tree.add(f"{SYMBOLS['warning']} {node_label}: Empty Parallel Block")
                return

            is_parallel_steps = all(isinstance(n, (Step, ApplyNode)) for n in node)
            if is_parallel_steps:
                parallel_type_str = "Steps"
                # Get names of parallel steps/applynodes
                step_names = []
                for n_idx, parallel_item in enumerate(node):
                    item_name = parallel_item.name or f"Unnamed ({parallel_item.id})"
                    item_emoji = getattr(parallel_item, 'emoji', SYMBOLS["step"] if isinstance(parallel_item, Step) else SYMBOLS["apply"])
                    step_names.append(f"{item_emoji} {item_name}")
                
                parallel_subtree = parent_tree.add(f"{node_label}: Parallel {parallel_type_str} ({len(node)} tasks)")
                for s_name in step_names:
                    parallel_subtree.add(s_name)

            else:  # Assuming parallel branches
                parallel_subtree = parent_tree.add(f"Step {index_prefix}: Parallel Branches ({len(node)} branches)")
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
            if not node:
                return f"{step_label}: Empty Parallel Block"
            
            is_parallel_steps = all(isinstance(n, (Step, ApplyNode)) for n in node)
            if is_parallel_steps:
                names = [n.name or n.id for n in node]
                return f"{step_label}: Parallel Steps: {', '.join(names)}"
            else:  # Assuming parallel branches
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
        console = Console()  # Added console instance
        
        if isinstance(node, Step):
            node_info = {
                "id": node.id,
                "name": node.name,
                "emoji": getattr(node, 'emoji', SYMBOLS["step"]),
                "type": "step"
            }
            
            if not current_records:
                console.print(f"{SYMBOLS['warning']}Skipping Step '{node.name}' - empty input", style=STYLE["warning"])
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
                console.print(f"{SYMBOLS['error']}Error in Step '{node.name}': {e}", style=STYLE["error"])
                raise
                
        elif isinstance(node, ApplyNode):
            node_info = {
                "id": node.id,
                "name": node.name,
                "emoji": SYMBOLS["apply"],
                "type": "apply"
            }
            
            if not current_records:
                console.print(f"{SYMBOLS['warning']} Skipping Apply '{node.name}' - empty input", style=STYLE["warning"])
                completed_steps[0] += 1
                progress.update(chain_task, completed=completed_steps[0])
                return []
                
            try:
                outputs = node.run(current_records)
                completed_steps[0] += 1
                progress.update(chain_task, completed=completed_steps[0])
                return outputs
            except Exception as e:
                console.print(f"{SYMBOLS['error']} Error executing Apply node '{node.name}': {e}", style=STYLE["error"])
                raise
                
        elif isinstance(node, list):
            if not current_records:
                console.print(f"{SYMBOLS['warning']}Skipping Parallel Block at step {i+1} - empty input", style=STYLE["warning"])
                completed_steps[0] += 1  # Still counts as one logical step
                progress.update(chain_task, completed=completed_steps[0])
                return []

            if not node:  # Empty list of nodes
                console.print(f"{SYMBOLS['warning']}Skipping empty parallel block at step {i+1}", style=STYLE["warning"])
                completed_steps[0] += 1  # Still counts as one logical step
                progress.update(chain_task, completed=completed_steps[0])
                return []

            is_parallel_steps = all(isinstance(n, (Step, ApplyNode)) for n in node)
            if is_parallel_steps:
                # This is parallel execution of steps/applynodes
                outputs = self._execute_parallel_steps(
                    node, current_records, writer, i, run_id, progress, chain_task, completed_steps
                )
                return outputs
            else:
                if not all(isinstance(n, Branch) for n in node):
                    console.print(f"{SYMBOLS['error']}Error: List at step {i+1} contains non-Branch items for parallel branch execution.", style=STYLE["error"])
                    completed_steps[0] += self._count_steps_in_node(node)
                    progress.update(chain_task, completed=completed_steps[0])
                    raise TypeError(f"List at step {i+1} must contain only Branch objects for parallel branch execution.")

                outputs = self._execute_branches(
                    node, current_records, writer, i, run_id, progress, chain_task, completed_steps
                )
                return outputs
            
        elif isinstance(node, Branch):
            node_info = {
                "id": node.name,
                "name": node.name, 
                "emoji": getattr(node, 'emoji', SYMBOLS["branch"]),
                "type": "branch"
            }
            console.print(f"{SYMBOLS['warning']} Branch execution logic needs review for '{node.name}'", style=STYLE["warning"])
            completed_steps[0] += 1
            progress.update(chain_task, completed=completed_steps[0])
            return current_records
            
        else:
            console.print(f"{SYMBOLS['warning']} Skipping unknown node type at index {i}: {type(node)}", style=STYLE["warning"])
            completed_steps[0] += 1
            progress.update(chain_task, completed=completed_steps[0])
            return []
            
        writer.add_node_to_graph({**node_info, "index": i})
        return []

    def _execute_parallel_steps(
        self,
        parallel_nodes: List[Union[Step, ApplyNode]],
        current_records: List[Any],
        writer: RunWriter,
        step_index: int,
        run_id: str,
        progress: Progress,
        chain_task: int,
        completed_steps: List[int]
    ) -> List[Dict[str, Any]]:
        """Executes a list of Step or ApplyNode objects in parallel on the same input records.
        The output for each input record is a dictionary of {node_id: output}.
        """
        console = Console()
        if not current_records:
            console.print(f"{SYMBOLS['warning']}Skipping Parallel Steps block at index {step_index} - empty input", style=STYLE["warning"])
            completed_steps[0] += 1
            progress.update(chain_task, completed=completed_steps[0])
            return []

        all_outputs_for_block = []

        parallel_block_id = f"parallel_steps_{step_index}"
        writer.add_node_to_graph({
            "id": parallel_block_id,
            "name": f"Parallel Steps Block {step_index + 1}",
            "type": "parallel_steps_block",
            "emoji": SYMBOLS["step"] + SYMBOLS["step"],
            "index": step_index,
            "children_ids": [n.id for n in parallel_nodes]
        })

        for record_idx, record in enumerate(current_records):
            record_output_dict = {}
            for node_idx, p_node in enumerate(parallel_nodes):
                node_specific_input = [record]
                node_output = None
                node_info = {
                    "id": p_node.id,
                    "name": p_node.name,
                    "type": "step" if isinstance(p_node, Step) else "apply",
                    "emoji": getattr(p_node, 'emoji', SYMBOLS["step"] if isinstance(p_node, Step) else SYMBOLS["apply"]),
                    "parent_block": parallel_block_id
                }

                try:
                    if isinstance(p_node, Step):
                        node_output_list = p_node.run(node_specific_input, run_id=run_id, step_index=step_index)
                        node_info["signature"] = p_node.signature()
                    elif isinstance(p_node, ApplyNode):
                        node_output_list = p_node.run(node_specific_input)
                    
                    if node_output_list:
                        node_output = node_output_list[0]
                        record_output_dict[p_node.id] = node_output.model_dump(mode='json') if isinstance(node_output, BaseModel) else node_output
                    else:
                        record_output_dict[p_node.id] = None
                        console.print(f"{SYMBOLS['warning']}Node {p_node.id} in parallel block returned no output for record {record_idx}", style=STYLE["warning"])

                except Exception as e:
                    console.print(f"{SYMBOLS['error']}Error in parallel node '{p_node.name}' (id: {p_node.id}) at step {step_index + 1}: {e}", style=STYLE["error"])
                    record_output_dict[p_node.id] = {"error": str(e)}
            
            all_outputs_for_block.append(record_output_dict)

        completed_steps[0] += 1
        progress.update(chain_task, completed=completed_steps[0])
        
        return all_outputs_for_block

    def _execute_branches(self, branches: List[Branch], current_records: List[Any], 
                         writer: RunWriter, i: int, run_id: str, 
                         progress: Progress, chain_task: int, completed_steps: List[int]) -> List[Dict[str, Any]]:
        """Execute a list of branches in parallel.
        The output for each input record is a dictionary of {branch_name: final_branch_output}.
        Returns a list of these dictionaries.
        """
        console = Console()
        aggregated_branch_outputs_per_record = [{} for _ in range(len(current_records))]

        if not current_records:
            console.print(f"{SYMBOLS['warning']}Skipping Parallel Branches block at index {i} - empty input", style=STYLE["warning"])
            num_steps_in_block = 0
            for br in branches:
                num_steps_in_block = max(num_steps_in_block, len(br.steps))
            completed_steps[0] += num_steps_in_block if num_steps_in_block > 0 else 1
            progress.update(chain_task, completed=completed_steps[0])
            return []

        branch_block_id = f"parallel_branches_{i}"
        writer.add_node_to_graph({
            "id": branch_block_id,
            "name": f"Parallel Branches Block {i + 1}",
            "type": "parallel_branches_block",
            "emoji": SYMBOLS["branch"] + SYMBOLS["branch"],
            "index": i,
            "children_ids": [br.name for br in branches]
        })
        
        max_steps_in_any_branch = 0

        for br_idx, br in enumerate(branches):
            if not isinstance(br, Branch):
                raise TypeError(f"Lists in Chain.steps must contain Branch objects (found {type(br)}) at step {i+1}")

            branch_input_records = [dict(r) if isinstance(r, BaseModel) else dict(r) for r in current_records]

            num_sub_steps = len(br.steps)
            max_steps_in_any_branch = max(max_steps_in_any_branch, num_sub_steps)

            writer.add_node_to_graph({
                "id": br.name,
                "name": br.name,
                "type": "branch",
                "emoji": getattr(br, 'emoji', SYMBOLS["branch"]),
                "parent_block": branch_block_id,
                "index": i,
                "branch_index": br_idx
            })

            current_branch_records = branch_input_records

            for sub_step_idx, sub_node in enumerate(br.steps):
                sub_step_label = f"Step {i+1}.{br_idx+1}.{sub_step_idx+1}"
                sub_node_name = f"'{sub_node.name}'" if hasattr(sub_node, 'name') and sub_node.name else f"({sub_node.id})"
                branch_progress_desc = (
                    f"Step {i+1}/{len(self.steps)}: Branch [u]{br.name}[/u] ({br_idx+1}/{len(branches)}) "
                    f"→ {sub_step_label} {sub_node_name}"
                )
                progress.update(chain_task, description=branch_progress_desc)

                node_outputs_for_branch_step = []
                try:
                    if isinstance(sub_node, Step):
                        node_outputs_for_branch_step = sub_node.run(current_branch_records, run_id=run_id, step_index=f"{i}.{br_idx}.{sub_step_idx}")
                        writer.add_node_to_graph({
                            "id": f"{br.name}.{sub_node.id}", "name": sub_node.name, "type": "step",
                            "branch": br.name, "signature": sub_node.signature(), "index": f"{i}.{br_idx}.{sub_step_idx}"
                        })
                    elif isinstance(sub_node, ApplyNode):
                        node_outputs_for_branch_step = sub_node.run(current_branch_records)
                        writer.add_node_to_graph({
                            "id": f"{br.name}.{sub_node.id}", "name": sub_node.name, "type": "apply",
                            "branch": br.name, "index": f"{i}.{br_idx}.{sub_step_idx}"
                        })
                    else:
                        raise TypeError(f"Unsupported node type {type(sub_node)} inside branch '{br.name}'")
                except Exception as e:
                    console.print(f"{SYMBOLS['error']}Error in Branch '{br.name}', Node '{getattr(sub_node, 'name', sub_node.id)}': {e}", style=STYLE["error"])
                    for record_agg_dict in aggregated_branch_outputs_per_record:
                        record_agg_dict[br.name] = {"error": f"Error in {sub_node.id}: {e}"}
                    current_branch_records = []
                    break

                writer.write_step(f"{br.name}.{sub_node.id}", node_outputs_for_branch_step)
                current_branch_records = node_outputs_for_branch_step

                if not current_branch_records:
                    console.print(f"{SYMBOLS['warning']}Branch '{br.name}', Node '{getattr(sub_node, 'name', sub_node.id)}' produced no output. Branch processing may halt or produce empty results.", style=STYLE["warning"])
                    break

                completed_steps[0] += 1
                progress.update(chain_task, completed=completed_steps[0])
            
            for record_idx, final_branch_output_for_record in enumerate(current_branch_records):
                if record_idx < len(aggregated_branch_outputs_per_record):
                    aggregated_branch_outputs_per_record[record_idx][br.name] = (
                        final_branch_output_for_record.model_dump(mode='json')
                        if isinstance(final_branch_output_for_record, BaseModel)
                        else final_branch_output_for_record
                    )
            for record_idx in range(len(current_branch_records), len(aggregated_branch_outputs_per_record)):
                 if br.name not in aggregated_branch_outputs_per_record[record_idx]:
                    aggregated_branch_outputs_per_record[record_idx][br.name] = None

        progress.update(chain_task, description=f"Step {i+1}/{len(self.steps)}: Parallel branches aggregated")

        return aggregated_branch_outputs_per_record
    
    def _count_steps_in_node(self, node: Any) -> int:
        """Helper to count effective steps for progress in case of error/skip for a complex node."""
        if isinstance(node, (Step, ApplyNode)):
            return 1
        elif isinstance(node, Branch):
            return len(node.steps) if node.steps else 1
        elif isinstance(node, list):
            is_parallel_steps = all(isinstance(n, (Step, ApplyNode)) for n in node)
            if is_parallel_steps:
                return 1
            else:
                max_branch_len = 0
                for br_node in node:
                    if isinstance(br_node, Branch):
                        max_branch_len = max(max_branch_len, len(br_node.steps) if br_node.steps else 1)
                return max_branch_len if max_branch_len > 0 else 1
        return 1

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
                    current_records = step_outputs
                    is_parallel_steps_type = all(isinstance(n, (Step, ApplyNode)) for n in node) if node else False
                    block_id = f"parallel_steps_output_{i}" if is_parallel_steps_type else f"parallel_branches_output_{i}"
                    if current_records:
                         writer.write_step(block_id, current_records)
                    
                elif step_outputs:
                    node_id_for_writer = node.id if hasattr(node, 'id') and node.id else \
                                       (node.name if hasattr(node, 'name') and node.name else f"unknown_node_{i}")
                    writer.write_step(node_id_for_writer, step_outputs)
                    current_records = step_outputs
                else:
                    if isinstance(node, (Step, ApplyNode)):
                        console.print(f"{SYMBOLS['warning']} Node {i+1} ('{getattr(node, 'name', 'Unnamed')}') produced None/empty output.", style=STYLE["warning"])
                    current_records = []
                    
                if not current_records and i < len(self.steps) - 1:
                    console.print(f"{SYMBOLS['warning']} No records to process. Stopping chain execution.", style=STYLE["warning"])
                    progress.update(chain_task, completed=total_steps, description="Chain stopped early.")
                    break

            progress.update(chain_task, description="Chain completed.", completed=total_steps)

        console.print(f"\n{SYMBOLS['chain']}Finalizing output...", style=STYLE["info"])
        ds_dict = writer.finalize(generate_flattened_output=generate_flattened_output)
        console.print(f"{SYMBOLS['success']}Chain '{self.name}' run finished. Output in {out_root.resolve()}", style="bold " + STYLE["success"])
        return ds_dict
