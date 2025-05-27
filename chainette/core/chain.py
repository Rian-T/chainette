from __future__ import annotations

"""Chain orchestration for Chainette."""

from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import gc
import json
import torch

from pydantic import BaseModel

from chainette.core.branch import Node, Branch
from chainette.core.step import Step
from chainette.io.writer import RunWriter
from chainette.engine.registry import EngineConfig, get_engine_config

__all__ = ["Chain"]


class Chain:
    def __init__(
        self,
        *,
        name: str,
        steps: List[Union[Node, List[Branch]]], # A step can now also be a single Branch
        emoji: str | None = None,
        batch_size: int = 1,
    ) -> None:
        self.name = name
        self.steps = steps
        self.emoji = emoji or ""
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #

    def run(
        self,
        inputs: List[BaseModel],
        *,
        output_dir: str | Path,
        fmt: str = "jsonl",
        generate_flattened_output: bool = True,
        max_lines_per_file: int = 1000,
        debug: bool = False,
    ):
        """Execute the chain and write results to *output_dir*."""
        writer = RunWriter(Path(output_dir), max_lines_per_file=max_lines_per_file, fmt=fmt)
        writer.set_chain_name(self.name)

        all_item_histories: List[Dict[str, Any]] = [{ "chain_input": inp } for inp in inputs]
        current_processing_objects: List[BaseModel] = list(inputs)
        
        active_step_engine_config: EngineConfig | None = None
        previous_step_node: Step | None = None 

        if debug: print(f"Starting chain '{self.name}' with {len(inputs)} items.")

        for node_idx, current_node_or_branch_list_item in enumerate(self.steps):
            if debug: print(f"Processing item #{node_idx} in chain.steps (type: {type(current_node_or_branch_list_item).__name__})")

            upcoming_step_engine_name: str | None = None
            is_current_item_a_step = isinstance(current_node_or_branch_list_item, Step)
            is_current_item_a_branch = isinstance(current_node_or_branch_list_item, Branch)
            is_current_item_a_list_of_branches = isinstance(current_node_or_branch_list_item, list)

            if is_current_item_a_step:
                upcoming_step_engine_name = current_node_or_branch_list_item.engine_name

            if active_step_engine_config:
                should_release = False
                if is_current_item_a_list_of_branches or is_current_item_a_branch:
                    should_release = True
                    if debug: print(f"Transitioning to a Branch or list of Branches, will release active engine '{active_step_engine_config.name}'.")
                elif is_current_item_a_step and upcoming_step_engine_name != active_step_engine_config.name:
                    should_release = True
                    if debug: print(f"Upcoming step uses different engine ('{upcoming_step_engine_name}' vs '{active_step_engine_config.name}'), will release active engine.")
                
                if should_release:
                    if debug: print(f"Releasing engine: {active_step_engine_config.name} (used by step: {previous_step_node.id if previous_step_node else 'N/A'})")
                    active_step_engine_config.release_engine()
                    if previous_step_node:
                        previous_step_node.engine = None
                        previous_step_node.tokenizer = None
                    active_step_engine_config = None
                    previous_step_node = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if debug: print("Engine released and garbage collected.")
            
            if is_current_item_a_list_of_branches:
                branches: List[Branch] = current_node_or_branch_list_item # It's a list of Branches
                if debug: print(f"Executing {len(branches)} branches from a list.")
                for br_idx, br in enumerate(branches):
                    if debug: print(f"Executing branch #{br_idx} ('{br.name}') from list.")
                    # Branches operate on copies of histories and current objects but don't modify them for the main chain flow directly
                    branch_item_histories = [hist.copy() for hist in all_item_histories]
                    br.execute(list(current_processing_objects), branch_item_histories, writer, debug=debug)
                # After a list of branches, the Chain's active engine concept is reset.
                active_step_engine_config = None
                previous_step_node = None
                if debug: print("Finished executing list of branches. Chain's active engine trackers reset.")
            
            elif is_current_item_a_branch:
                branch_node: Branch = current_node_or_branch_list_item
                if debug: print(f"Executing single Branch node: {branch_node.id} ('{branch_node.name}')")
                branch_item_histories = [hist.copy() for hist in all_item_histories]
                branch_node.execute(list(current_processing_objects), branch_item_histories, writer, debug=debug)
                active_step_engine_config = None
                previous_step_node = None
                if debug: print(f"Node {branch_node.id} (Branch) finished. Chain's active engine trackers reset.")
            
            else: # Must be a Step or Apply node
                node: Node = current_node_or_branch_list_item 
                if debug: print(f"Executing single node: {node.id} (type: {type(node).__name__})")
                
                # Apply and Step nodes return updated objects and histories
                current_processing_objects, all_item_histories = node.execute(
                    current_processing_objects, all_item_histories, writer, debug=debug
                )

                if isinstance(node, Step):
                    active_step_engine_config = get_engine_config(node.engine_name)
                    previous_step_node = node
                    if debug: print(f"Node {node.id} (Step) finished. Active engine is now '{active_step_engine_config.name}'.")
                # If node is Apply, active_step_engine_config and previous_step_node remain as they were from the preceding Step.
        
        if active_step_engine_config:
            if debug: print(f"End of chain. Releasing final active engine: {active_step_engine_config.name} (used by step: {previous_step_node.id if previous_step_node else 'N/A'})")
            active_step_engine_config.release_engine()
            if previous_step_node:
                previous_step_node.engine = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if debug: print("Final engine released and garbage collected.")
            
        return writer.finalize(generate_flattened_output=generate_flattened_output)
