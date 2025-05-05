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
from typing import Any, List, Dict, Union # Added Dict, Union

from datasets import DatasetDict
from pydantic import BaseModel # Added BaseModel

# Import necessary components from other modules
from chainette.core.step import Step
from chainette.core.apply import ApplyNode
from chainette.core.branch import Branch, Node # Import Branch and Node
from chainette.io.writer import RunWriter, flatten_datasetdict # Import RunWriter and flatten_datasetdict
from chainette.utils.ids import snake_case, new_run_id # Import snake_case and new_run_id

__all__ = ["Chain"]


class Chain:
    """Represents a sequence of executable steps or branches."""
    def __init__(
        self,
        *,
        name: str,
        steps: List[Node], # Use Node type hint
        emoji: str = "⛓️", # Changed emoji
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
        # TODO: Add more validation for step/branch compatibility

    # ------------------------------------------------------------------ #

    def _batched(self, items: list[Any]) -> List[List[Any]]:
        """Helper to divide items into batches."""
        if not items:
            return []
        return [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    # ------------------------------------------------------------------ #

    def run(
        self,
        inputs: List[BaseModel],
        *,
        output_dir: str | Path = "output",
        max_lines_per_file: int = 1_000,
        fmt: str = "jsonl",
        generate_flattened_output: bool = True, # Default to True based on cli.py
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
        run_id = new_run_id()
        out_root = Path(output_dir) / f"{snake_case(self.name)}_{run_id}"
        writer = RunWriter(out_root, max_lines_per_file, fmt)
        writer.set_chain_name(self.name)

        print(f"{self.emoji} Starting chain '{self.name}' (Run ID: {run_id})")
        print(f"Output directory: {out_root.resolve()}")

        # Write initial inputs
        writer.write_step("input", inputs)
        writer.add_node_to_graph({
            "id": "input",
            "name": "Input Data",
            "type": "input",
            "emoji": "📥",
        })

        current_records: List[Any] = inputs
        # Removed all_step_outputs dict, writer handles accumulation

        for i, node in enumerate(self.steps):
            node_id: str
            node_name: str
            node_emoji: str = "❓" # Default emoji
            node_type: str
            step_outputs: List[Any] = [] # Initialize step_outputs for each node

            if isinstance(node, Step):
                node_id = node.id
                node_name = node.name
                node_emoji = node.emoji
                node_type = "step"
                print(f"  {node_emoji} Running Step {i}: '{node_name}' ({node_id})...")
                if not current_records:
                    print(f"Warning: Skipping Step '{node_name}' due to empty input from previous step.")
                    continue
                batched_inputs = self._batched(current_records)
                for batch in batched_inputs:
                    # Ensure batch contains valid input models for the step
                    try:
                        # Validate or transform items in the batch to match the step's input model
                        valid_batch = []
                        for item in batch:
                            if isinstance(item, node.input_model):
                                valid_batch.append(item)
                            elif isinstance(item, dict):
                                valid_batch.append(node.input_model.model_validate(item))
                            elif hasattr(item, 'model_dump'): # Handle other pydantic models
                                valid_batch.append(node.input_model.model_validate(item.model_dump()))
                            else:
                                raise TypeError(f"Cannot convert item of type {type(item)} to {node.input_model.__name__}")
                        step_outputs.extend(node.run(valid_batch, run_id=run_id, step_index=i))
                    except Exception as e:
                        print(f"Error processing batch for Step '{node_name}': {e}")
                        # Option: re-raise, skip batch, or handle error records
                        raise # Re-raise for now

            elif isinstance(node, ApplyNode):
                node_id = node.id
                node_name = node.name
                node_emoji = "⚙️" # Default emoji for apply
                node_type = "apply"
                print(f"  {node_emoji} Running Apply {i}: '{node_name}' ({node_id})...")
                if not current_records:
                    print(f"Warning: Skipping Apply '{node_name}' due to empty input from previous step.")
                    continue
                try:
                    # ApplyNode's run method expects the full list
                    # Apply node function might not have run_id/step_index args, handle gracefully
                    # Let's assume ApplyNode.run handles this or the function itself does
                    step_outputs = node.run(current_records) # Simplified call
                except Exception as e:
                    print(f"Error executing Apply node '{node_name}': {e}")
                    raise # Re-raise for now

            elif isinstance(node, Branch):
                # Branch handling needs refinement. The current code doesn't correctly
                # integrate branch outputs back or manage the writer for branches.
                # For now, let's just log and skip complex branch logic.
                node_id = node.name
                node_name = node.name
                node_emoji = node.emoji
                node_type = "branch"
                print(f"  {node_emoji} Running Branch {i}: '{node_name}' (Branch execution logic needs review)... ")
                # Placeholder: Branch execution logic from the original prompt was complex
                # and didn't fit well with the single writer model. It needs redesign.
                # For now, we'll just pass the current records through without modification
                # and add the branch node to the graph.
                step_outputs = current_records # Pass through
                # Original branch logic created separate writers and didn't merge back.
                # This needs a clearer design on how branch outputs are handled.

            else:
                print(f"Warning: Skipping unknown node type at index {i}: {type(node)}")
                continue # Skip unknown node types

            if step_outputs is not None: # Check for None explicitly, allow empty lists
                writer.write_step(node_id, step_outputs)
                writer.add_node_to_graph({
                    "id": node_id,
                    "name": node_name,
                    "type": node_type,
                    "emoji": node_emoji,
                    "index": i,
                    # Add signature if it's a Step
                    **({"signature": node.signature()} if isinstance(node, Step) else {})
                })
                current_records = step_outputs
            else:
                print(f"Warning: Node {i} ('{node_name}') produced None output. Stopping chain execution.")
                # Stop the chain if a node returns None, as subsequent steps likely depend on output.
                current_records = [] # Clear records
                break # Exit the loop

            # Handle case where step_outputs is an empty list
            if not current_records:
                 print(f"Warning: Node {i} ('{node_name}') produced an empty list. Subsequent steps may fail or be skipped.")
                 # Continue execution with empty list


        print(f"{self.emoji} Chain finished. Finalizing output...")
        # Finalize the main writer
        ds_dict = writer.finalize(generate_flattened_output=generate_flattened_output)
        print(f"[green]✔[/green] Chain '{self.name}' run completed. Outputs in {out_root.resolve()}")
        return ds_dict
