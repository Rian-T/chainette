# 12 – Flagship Incident Pipeline 🚀

A comprehensive demo showing Chainette's full power on a realistic SRE / DevOps
workflow.

Pipeline overview:

1. **Retrieve Similar Incidents** (ApplyNode)
2. **Summarise & Recommend** (LLM Step)
3. **Translate** summary into 🇪🇸 Spanish and 🇫🇷 French (parallel Branches)
4. **Grade** summary quality 1–10 (LLM Judge)

Run it:
```bash
poetry run chainette run examples/12_flagship/chain.py flagship_chain \
    examples/12_flagship/inputs.jsonl _tmp_flagship_out --quiet --no-icons
```

Outputs written per step + flattened view.

> Requires `OPENAI_API_KEY`. 