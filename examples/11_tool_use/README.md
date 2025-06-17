# 11 – Tool Use Demo

This example shows how Chainette can integrate **tool use** in a pipeline:

1. An **LLM step** inspects the user's question and outputs JSON indicating
   which tool to call (`calculator` or `weather`) and the required arguments.
2. A pure-python **Apply node** executes the selected tool and returns the
   final answer.

The weather tool is stubbed to avoid external dependencies; the calculator
supports basic arithmetic expressions via a very limited `eval` after a
character whitelist check.

Run the example:

```bash
poetry run chainette run examples/11_tool_use/chain.py tool_use_chain \
    examples/11_tool_use/inputs.jsonl _tmp_tool_use_out --quiet --no-icons
```

Expected flattened output (example):

```json
{"answer": "56"}
{"answer": "It's sunny in Paris ☀️ (stubbed)"}
```

> Requires `OPENAI_API_KEY` in the environment. 