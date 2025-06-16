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

# Optional pyarrow import for Parquet support – keeps dependency lightweight
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    _HAS_PARQUET = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_PARQUET = False

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

        # For parquet buffered writing
        if self.fmt == "parquet":
            self._buffers: Dict[str, List[Dict[str, Any]]] = {}

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
            if _HAS_PARQUET:
                return pq.ParquetWriter(str(fp), schema=Features({}).arrow_schema)  # type: ignore[arg-type]
            else:
                raise RuntimeError("Parquet support not available")

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
        if self.fmt == "jsonl":
            handle = self._handle_for(split)

            for rec in records:
                row = rec.model_dump(mode="json") if isinstance(rec, BaseModel) else rec
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                if self.flattened:
                    self._write_flat_row(split, row)

                self._line_counts[split] += 1
                if self._line_counts[split] >= self.max_lines:
                    handle.close()
                    self._handles.pop(split)
                    self._line_counts[split] = 0
                    # open new file for subsequent rows in current batch
                    handle = self._handle_for(split)

        elif self.fmt == "parquet":
            if not _HAS_PARQUET:
                raise RuntimeError("pyarrow not installed – install to use parquet output")
            buf = self._buffers.setdefault(split, [])  # type: ignore[attr-defined]
            for rec in records:
                row = rec.model_dump(mode="json") if isinstance(rec, BaseModel) else rec
                buf.append(row)
                if self.flattened:
                    self._write_flat_row(split, row)

                if len(buf) >= self.max_lines:
                    self._flush_parquet_buffer(split)

        else:
            raise ValueError(f"Unsupported format {self.fmt}")

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

        if self.fmt == "parquet" and _HAS_PARQUET:
            for split in list(getattr(self, "_buffers", {}).keys()):
                self._flush_parquet_buffer(split)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ------------------------------------------------------------------ #
    # Compatibility stubs with legacy RunWriter API
    # ------------------------------------------------------------------ #

    def add_node_to_graph(self, node_info: dict[str, Any]):  # noqa: D401
        """Legacy no-op: StreamWriter does not store the exec graph yet."""
        # TODO: persist lightweight execution graph as line-delimited JSON.
        return None

    def set_chain_name(self, name: str):  # noqa: D401
        """Legacy no-op (kept for interface parity)."""
        return None

    # ------------------------------------------------------------------ #
    # Parquet helpers
    # ------------------------------------------------------------------ #

    def _flush_parquet_buffer(self, split: str):
        if not _HAS_PARQUET:
            return
        buf: List[Dict[str, Any]] = self._buffers.get(split, [])  # type: ignore[attr-defined]
        if not buf:
            return
        idx = self._file_numbers.setdefault(split, count(0))
        file_idx = next(idx)
        dir_ = self.root / split
        dir_.mkdir(parents=True, exist_ok=True)
        fp = dir_ / f"{file_idx:03d}.parquet"
        table = pa.Table.from_pylist(buf)
        pq.write_table(table, fp)
        buf.clear() 