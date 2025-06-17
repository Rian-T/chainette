# 01 – Sentiment Classifier (Pure LLM Pipeline Starter)

A hello-world Chainette demo.  It ingests short product reviews and returns a
JSON sentiment label (`positive` / `negative` / `neutral`).  For self-contained
execution we ship a **heuristic Python classifier**, but you can switch it to an
OpenAI `Step` by uncommenting the lines in `chain.py`.

Run
```bash
poetry run chainette run examples/01_sentiment/chain.py sentiment_chain examples/01_sentiment/inputs.jsonl _out_sentiment
```

Expected output (truncated):
```
pos,neg
``` 