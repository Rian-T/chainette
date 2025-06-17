# 08 – YAML Customer Support Bot

Illustrates a pipeline declared purely in YAML. Steps are defined in
`examples/08_chatbot_yaml/steps.py` and referenced via module path.

Run:
```bash
poetry run chainette run-yaml examples/08_chatbot_yaml/chain.yml _out_chatbot \
    --symbols-module examples.08_chatbot_yaml.steps --quiet --no-icons
``` 