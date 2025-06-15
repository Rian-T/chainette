# Contributing to Chainette

Thank you for considering contributing!  Chainette strives to remain a **tiny, easy-to-read** codebase.  Please follow these simple rules:

1. **Keep it small**  – Aim for modules < 150 LOC.  Each public function should be readable without scrolling.
2. **Type everything** – Public APIs must be fully type-annotated.
3. **Explicit is better** – No hidden magic, no globals beyond the engine registry/pool.
4. **Stay dependency-light** – New external packages need strong justification.
5. **Follow the folder layout**
   * `chainette/core/` – Execution primitives (Node, Step, Branch, Graph, Executor)
   * `chainette/engine/` – Engine config & pooling
   * `chainette/utils/` – Pure helpers (prompt, ids, json_schema…)
6. **Tests first** – Extend `tests/` or the integration script (`examples/ollama_gemma_features.py`) whenever you add a new feature.
7. **Update docs** – After a change, update `README.md`, `llm.txt` and `ELEGANCE_PLAN.md` snippet.
8. **Run the checklist**  – Execute:
   ```bash
   poetry run chainette run examples/ollama_gemma_features.py full_chain inputs2.jsonl _tmp_run_<n>
   ```
   Ensure DEBUG output is correct and inspect artefacts under `_tmp_run_<n>/`.

Happy hacking! 