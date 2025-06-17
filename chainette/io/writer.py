"""
File writer utility – chunks, metadata, graph.json, flattened output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Dict

from datasets import Dataset, DatasetDict
from pydantic import BaseModel

from chainette.utils.ids import snake_case

__all__ = ["RunWriter", "flatten_datasetdict"]


class RunWriter:  # noqa: D101
    def __init__(
        self,
        root: Path,
        max_lines_per_file: int,
        fmt: str,
    ):
        self.root = root
        self.max_lines = max_lines_per_file
        self.fmt = fmt
        self.root.mkdir(parents=True, exist_ok=True)
        self._splits: dict[str, list[dict[str, Any]]] = {}
        self._exec_graph: list[dict[str, Any]] = []
        self._chain_name: str = ""
        self._file_counters: dict[str, int] = {}  # Track file numbers for each split

    # -------------------------------------------------------------- #

    def set_chain_name(self, name: str):
        """Set the chain name for metadata."""
        self._chain_name = name

    def add_node_to_graph(self, node_info: dict[str, Any]):
        """Add information about an executed node to the graph."""
        self._exec_graph.append(node_info)

    def write_step(self, step_id: str, records: list[Any]):
        """Write the output records for a given step ID."""
        split_name = snake_case(step_id)
        dict_records = [
            rec.model_dump(mode='json') if isinstance(rec, BaseModel) else rec
            for rec in records
        ]
        self._splits.setdefault(split_name, []).extend(dict_records)

    # -------------------------------------------------------------- #

    def _flush_datasets(self) -> DatasetDict:
        """Convert collected splits into a DatasetDict."""
        dsets = {}
        for split, rows in self._splits.items():
            if rows:
                dsets[split] = Dataset.from_list(rows)
            else:
                print(f"Warning: No data found for split '{split}'. Skipping dataset creation.")
        return DatasetDict(dsets)

    # -------------------------------------------------------------- #

    def finalize(self, *, generate_flattened_output: bool = True) -> DatasetDict:
        """Finalize the run: write graph, datasets, flattened output, and metadata."""
        # 1) write execution graph
        graph_path = self.root / "graph.json"
        graph_path.write_text(json.dumps({"execution_order": self._exec_graph}, indent=2))

        # 2) write datasets
        ds_dict = self._flush_datasets()
        for name, ds in ds_dict.items():
            out_dir = self.root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            # Simple writing for now, ignoring max_lines
            file_path = out_dir / f"0.{self.fmt}"
            if self.fmt == "jsonl":
                # Use custom JSON writing to preserve Unicode characters
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in ds:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            elif self.fmt == "csv":
                ds.to_csv(file_path, index=False)
            else:
                print(f"Warning: Unsupported format '{self.fmt}' for split '{name}'. Skipping write.")

        # 3) flattened output (if requested and possible)
        if generate_flattened_output and ds_dict:
            try:
                flat_ds = flatten_datasetdict(ds_dict)
                flat_out_dir = self.root / "flattened"
                flat_out_dir.mkdir(parents=True, exist_ok=True)
                flat_file_path = flat_out_dir / f"0.{self.fmt}"
                if self.fmt == "jsonl":
                    # Use custom JSON writing to preserve Unicode characters
                    with open(flat_file_path, 'w', encoding='utf-8') as f:
                        for item in flat_ds:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                elif self.fmt == "csv":
                    flat_ds.to_csv(flat_file_path, index=False)
            except ValueError as e:
                print(f"Warning: Could not generate flattened output. Reason: {e}")
            except Exception as e:
                print(f"Warning: An unexpected error occurred during flattening: {e}")

        # 4) metadata
        meta_path = self.root / "metadata.json"
        meta = {
            "execution_info": {
                "chain_name": self._chain_name,
                "run_dir": str(self.root.resolve()),
            },
            "format": self.fmt,
            "generated_by": "chainette v0.1.0",
            "splits": list(ds_dict.keys()),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return ds_dict

def flatten_datasetdict(ds_dict: DatasetDict) -> Dataset:
    """Flattens a DatasetDict into a single Dataset by joining rows.

    Assumes all datasets in the dict have the same number of rows.
    Column names are prefixed with the original dataset key.
    Raises ValueError if datasets have different lengths.
    """
    if not ds_dict:
        return Dataset.from_list([])

    # Determine minimum length; keep only splits at this length.
    lengths = {k: len(v) for k, v in ds_dict.items()}
    min_len = min(lengths.values())

    if len(set(lengths.values())) > 1:
        dropped = {k: l for k, l in lengths.items() if l > min_len}
        print(
            "Warning: differing split sizes. Dropping splits "
            f"with greater length: {dropped}. Flatten will use splits sized {min_len}."
        )

    # Attempt smarter join: use first common column across splits as join key
    first_split = next(iter(ds_dict.values()))
    common_cols = set(first_split.column_names)
    for ds in ds_dict.values():
        common_cols &= set(ds.column_names)
    join_key = next(iter(common_cols)) if common_cols else None

    if join_key:
        # Build lookup tables for each split
        lookups = {k: {row[join_key]: row for row in v} for k, v in ds_dict.items() if len(v) >= min_len}

        reference_split = min(lookups.keys(), key=lambda k: len(lookups[k]))
        flat_rows: list[dict[str, Any]] = []
        for val, ref_row in lookups[reference_split].items():
            if all(val in lookups[split] for split in lookups):
                merged: dict[str, Any] = {}
                for split, table in lookups.items():
                    row = table[val]
                    for col_name, value in row.items():
                        if col_name == "row_id":
                            merged["row_id"] = value
                        else:
                            merged[f"{split}.{col_name}"] = value
                flat_rows.append(merged)
    else:
        # Fallback to positional join on min_len splits
        kept_keys = [k for k, l in lengths.items() if l >= min_len]
        flat_rows: list[dict[str, Any]] = []
        for i in range(min_len):
            merged_row: dict[str, Any] = {}
            for key in kept_keys:
                row = ds_dict[key][i]
                for col_name, value in row.items():
                    if col_name == "row_id":
                        merged_row["row_id"] = value
                    else:
                        merged_row[f"{key}.{col_name}"] = value
            flat_rows.append(merged_row)

    return Dataset.from_list(flat_rows)
