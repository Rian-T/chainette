# 13 – French Clinical Notes (Pancreatic Cancer)

End-to-end demo:

1. **fetch_inputs.py** – pulls 100 French clinical cases from the HF dataset
   `rntc/open-clinical-cases-pubmed-comet`, split `fr_translated`, and writes
   `inputs.jsonl`.
2. **pancreas_chain** (chain.py)
   • Step 1 – LLM classifier (`o4-mini`) flags pancreatic-cancer cases.
   • Step 2 – Apply node filters non-relevant cases.
   • Step 3 – LLM reformulates text into a realistic French hospital note.

Run:
```bash
# One-off: download dataset (needs `datasets`)
python examples/13_pancreas_note_fr/fetch_inputs.py

# Execute chain
poetry run chainette run examples/13_pancreas_note_fr/chain.py pancreas_chain \
    examples/13_pancreas_note_fr/inputs.jsonl _out_pancreas_notes --quiet --no-icons
```

Outputs will appear under `_out_pancreas_notes/`, with a flattened JSONL for quick inspection. 