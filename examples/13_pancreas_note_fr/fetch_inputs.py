"""Fetch first 100 rows from the fr_translated split of the open-clinical-cases dataset
and write as inputs.jsonl (field: text). Run once before executing the chain.
"""

from datasets import load_dataset
import json
from pathlib import Path

OUT = Path(__file__).parent / "inputs.jsonl"

ds = load_dataset("rntc/open-clinical-cases-pubmed-comet", split="fr_translated", streaming=False)

subset = ds.select(range(10000))
with OUT.open("w", encoding="utf-8") as f:
    for i, row in enumerate(subset):
        f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    print(f"Wrote {i+1} rows to {OUT}") 