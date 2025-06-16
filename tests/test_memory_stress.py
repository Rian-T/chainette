import pytest, os, psutil, json, tempfile
from pathlib import Path

pytestmark = pytest.mark.skipif(os.getenv("STRESS") != "1", reason="stress test disabled by default")


def test_memory_big_run():
    from chainette.examples.runner.huge_batch_demo import demo_chain, _ensure_inputs, Number
    inp = Path(tempfile.gettempdir()) / "stress_inputs.jsonl"
    _ensure_inputs(inp, n=200000)  # 200k rows

    from chainette.io.stream_writer import StreamWriter

    writer = StreamWriter(Path(tempfile.gettempdir()) / "stress_out", max_lines_per_file=5000)
    import json
    inputs = [Number(value=i) for i in range(200000)]

    p = psutil.Process(os.getpid())
    before = p.memory_info().rss / 1e6
    demo_chain.run(inputs, writer=writer)
    after = p.memory_info().rss / 1e6
    # We allow up to 300 MB increase for 200k rows
    assert (after - before) < 300 