# Branching Example – Parallel Branches & Join

Demonstrates how to run two parallel branches that increment and decrement a
number, then merge their outputs back into the main history using
`JoinBranch`.

Files:
* `chain.py`
* `inputs.jsonl`

Run:
```bash
poetry run chainette run examples/branching/chain.py branching_chain inputs.jsonl _out_branching
``` 