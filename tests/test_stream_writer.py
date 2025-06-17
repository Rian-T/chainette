import os
import json
import tempfile
from pathlib import Path

import pytest

from chainette.io.stream_writer import StreamWriter, _HAS_PARQUET
from pydantic import BaseModel


class Dummy(BaseModel):
    value: int


def _write_records(tmp_path: Path, fmt: str = "jsonl"):
    writer = StreamWriter(tmp_path, max_lines_per_file=5, fmt=fmt)
    for i in range(12):
        writer.write_step("dummy", [Dummy(value=i)])
    writer.close()
    return tmp_path


def test_stream_writer_jsonl(tmp_path):
    out_dir = _write_records(tmp_path, "jsonl")
    files = list((out_dir / "dummy").glob("*.jsonl"))
    assert len(files) == 3  # 0,1,2 files due to max_lines=5
    total = 0
    for fp in files:
        total += sum(1 for _ in fp.open())
    assert total == 12


@pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")

def test_stream_writer_parquet(tmp_path):
    out_dir = _write_records(tmp_path, "parquet")
    files = list((out_dir / "dummy").glob("*.parquet"))
    import pyarrow.parquet as pq  # type: ignore

    counts = sum(pq.read_table(fp).num_rows for fp in files)
    assert counts == 12 