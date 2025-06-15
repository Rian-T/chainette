from __future__ import annotations
"""Streaming file writer for massive Chainette runs.

Compared to `RunWriter`, this implementation **never keeps full splits in
memory** – rows are flushed to disk as soon as they arrive.  Files roll
when `max_lines_per_file` is reached, producing `000.jsonl`,
`001.jsonl`, … per split.  Flattened rows are streamed too so memory
remains O(batch).

Usage (within Executor):
```python
writer = StreamWriter(out_dir, max_lines_per_file=1_000, fmt="jsonl")
writer.write_step("my_step", outputs)  # list[BaseModel] | list[dict]
...
writer.close()
```
"""

from pathlib import Path
from typing import Any, Dict, IO, List
import json
from itertools import count

from datasets import Features  # type: ignore
from pydantic import BaseModel

from chainette.utils.ids import snake_case
from chainette.utils.events import publish, BatchFinished

__all__ = ["StreamWriter"]


class StreamWriter:  # noqa: D101
    def __init__(
        self,
        root: Path,
        *,
        max_lines_per_file: int = 10_000,
        fmt: str = "jsonl",  # jsonl | parquet
        flattened: bool = True,
    ):  # noqa: D401
        self.root = root
        self.max_lines = max_lines_per_file
        self.fmt = fmt.lower()
        self.flattened = flattened

        self.root.mkdir(parents=True, exist_ok=True)

        # Per-split bookkeeping
        self._handles: Dict[str, IO] = {}
        self._line_counts: Dict[str, int] = {}
        self._file_numbers = {k: count(0) for k in []}  # lazily created

        # For flattened streaming
        if self.flattened:
            flat_dir = self.root / "flattened"
            flat_dir.mkdir(parents=True, exist_ok=True)
            self._flat_handle: IO | None = None
            self._flat_lines: int = 0
            self._flat_num = count(0)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _open_handle(self, split: str) -> IO:
        """Return file handle for *split* respecting rolling policy."""
        num = self._file_numbers.setdefault(split, count(0))
        idx = next(num)
        dir_ = self.root / split
        dir_.mkdir(parents=True, exist_ok=True)
        suffix = ".jsonl" if self.fmt == "jsonl" else ".parquet"
        fp = dir_ / f"{idx:03d}{suffix}"
        if self.fmt == "jsonl":
            return open(fp, "w", encoding="utf-8")
        else:  # parquet deferred – for now create placeholder handle
            import pyarrow.parquet as pq  # type: ignore

            return pq.ParquetWriter(str(fp), schema=Features({}).arrow_schema)  # type: ignore[arg-type]

    def _handle_for(self, split: str) -> IO:
        h = self._handles.get(split)
        if h is None:
            h = self._open_handle(split)
            self._handles[split] = h
            self._line_counts[split] = 0
        return h

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def write_step(self, step_id: str, records: List[Any]):  # noqa: D401
        """Stream *records* (list[dict|BaseModel]) under *step_id* split."""
        if not records:
            return
        split = snake_case(step_id)
        handle = self._handle_for(split)

        for rec in records:
            row = rec.model_dump(mode="json") if isinstance(rec, BaseModel) else rec
            if self.fmt == "jsonl":
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                raise NotImplementedError("parquet not yet implemented in StreamWriter")

            # Flatten on the fly if enabled
            if self.flattened:
                self._write_flat_row(split, row)

            # Roll file if needed
            self._line_counts[split] += 1
            if self._line_counts[split] >= self.max_lines:
                handle.close()
                self._handles.pop(split)
                # next call will open a new file
                self._line_counts[split] = 0

        publish(BatchFinished(step_id=step_id, batch_no=-1, count=len(records)))

    def _write_flat_row(self, split: str, row: Dict[str, Any]):
        """Write flattened *row* to flat file handle."""
        if self._flat_handle is None:
            idx = next(self._flat_num)
            flat_fp = self.root / "flattened" / f"{idx:03d}.jsonl"
            self._flat_handle = open(flat_fp, "w", encoding="utf-8")
            self._flat_lines = 0
        merged = {f"{split}.{k}": v for k, v in row.items()}
        self._flat_handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
        self._flat_lines += 1
        if self._flat_lines >= self.max_lines:
            self._flat_handle.close()
            self._flat_handle = None

    # ------------------------------------------------------------------ #

    def close(self):  # noqa: D401
        """Close all open file handles."""
        for h in self._handles.values():
            h.close()
        self._handles.clear()
        if self._flat_handle is not None:
            self._flat_handle.close()
            self._flat_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close() 