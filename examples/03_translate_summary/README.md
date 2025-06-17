# 03 – Translate & Summary (Branch + Join)

Demonstrates parallel translation into French, Spanish, and German, then a
downstream summariser step.
(Current offline version produces dummy translations.)

Run:
```bash
poetry run chainette run examples/03_translate_summary/chain.py translate_chain \
    examples/03_translate_summary/inputs.jsonl _out_trans --quiet --no-icons
``` 