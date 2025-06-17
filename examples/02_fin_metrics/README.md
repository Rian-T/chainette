# 02 – Financial Metrics Extractor

Extracts `company`, `metric`, and `value` fields from short earnings blurbs.
Heuristic implementation runs offline; swap to OpenAI by uncommenting LLM block
in `chain.py`.

Run:
```bash
poetry run chainette run examples/02_fin_metrics/chain.py fin_metrics_chain \
    examples/02_fin_metrics/inputs.jsonl _out_fin --quiet --no-icons
```

Files:

* `chain.py` – defines the `fin_metrics_chain` object.
* `inputs.jsonl` – sample inputs (line-delimited JSON).
* `README.md` – this file.

Expected DAG:

```
[ChainInput] → [extract]
``` 