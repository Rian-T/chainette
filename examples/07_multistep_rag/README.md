# 07 – Multi-step RAG Demo

Shows a minimal Retrieval-Augmented Generation pipeline:
1. `retrieve` ApplyNode returns a short context string from an in-memory dict.
2. OpenAI `answer` Step uses the context to answer the question.

Run:
```bash
poetry run chainette run examples/07_multistep_rag/chain.py rag_chain \
    examples/07_multistep_rag/inputs.jsonl _out_rag --quiet --no-icons
``` 