# Pure Python Example – Apply Node

Shows how to drop a normal Python function into a Chainette pipeline via the
`apply()` helper.  No LLM calls involved.

Run:
```bash
poetry run chainette run examples/pure_python/chain.py py_chain inputs.jsonl _out_py
``` 