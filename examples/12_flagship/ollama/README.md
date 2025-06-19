# 12 – Flagship Incident Pipeline (Ollama) 🚀

This version of the flagship incident-management demo uses a **local Ollama** server
running the `gemma3:4b` model.

Make sure you have [Ollama](https://ollama.ai) installed and `ollama serve` is
running. The first run will pull the `gemma3:4b` model automatically (≈4 GB).

Run the pipeline:

```bash
poetry run chainette run examples/12_flagship/ollama/chain.py flagship_chain_ollama \
    examples/12_flagship/ollama/inputs.jsonl _tmp_flagship_ollama_out --quiet --no-icons
```

The command writes per-step outputs plus a flattened dataset to the specified
output directory.

> No external API keys required – everything runs locally via Ollama. 