# Chainette Codebase Cleanup & Example Overhaul Plan

> Branch: `code-cleanup`

This living document outlines the **one-item-at-a-time** roadmap to declutter the Chainette repository, remove dead code paths, and publish a concise, professional example suite powered exclusively by the OpenAI backend.  The refactor honours Chainette's guiding philosophy (see `llm.txt`): *tiny, type-safe, dependency-light and reproducible*.

-----------------------------------------------------------------------------
## 0 – Context Snapshot (June 2025)

*Before* (`main`):
• Legacy helper modules kept for unreleased retro-compatibility (e.g. `utils.logging_v2.py`).
• Example files scattered under `examples/`, many redundant or outdated, some still referencing in-process vLLM.
• Comments include TODOs, experimental notes, and pre-release chatter.
• LOC creeping up – > 10 k lines across library + examples.

*Target* (`code-cleanup`):
• ≤ 6 k LOC including tests & examples.
• Zero abandoned modules; public API unchanged.
• One coherent **examples/** tree showcasing every feature using the OpenAI backend.
• Clear, production-ready comments & docstrings only.

-----------------------------------------------------------------------------
## 1 – Design Goals

G1. **Preserve functionality** – No public API breaks; all unit tests must still pass.
G2. **Minimise surface area** – Delete unused modules, collapse trivial wrappers, and merge near-duplicate helpers.
G3. **Philosophy first** – Honour the principles in `llm.txt` (strict typing, small foot-print, reproducibility, composability).
G4. **Showcase via OpenAI** – A curated example suite demonstrating:
   • Basic single-step chain.
   • Multi-step data-flow with history templating.
   • Apply / pure-python transform node.
   • Branch & Join mechanics.
   • Batch execution & RunWriter outputs.
G5. **Documentation-as-code** – Each example directory contains a *README.md* with an ASCII DAG diagram and expected sample output.

-----------------------------------------------------------------------------
## 2 – Scope of Cleanup

1. **Delete legacy modules**
   • `utils.logging_v2.py` *(already removed in git status)*.
   • Any `__pycache__` or orphaned tmp dirs under version control.
2. **Consolidate utils**
   • Merge `utils.logging_v3` → rename to `utils.logging` (single source).
   • Fold tiny helpers (e.g. `utils.constants`) directly into their primary modules when <10 LOC.
3. **Remove vLLM-specific test paths** not relevant after `ENGINE_API` migration.
4. **Trim comments**
   • Delete pre-release TODO blocks and noisy print debug.
   • Keep high-signal docstrings only.
5. **Re-organise `examples/`**
   A curated catalogue of **10 progressively richer, real-world demos** (OpenAI backend only).  Folder names are prefixed with a two-digit index to make ordering explicit:

   | Folder | Use-case | Highlights |
   |--------|---------|------------|
   | `01_sentiment/` | JSON sentiment classification | Single Step |
   | `02_fin_metrics/` | Company metric extractor (earnings blurbs) | Schema enforcement |
   | `03_translate_summary/` | Translate EN→FR/ES/DE then summarise | Branch + Join |
   | `04_branch_math/` | Increment & Decrement in parallel | Pure-python functions + Join |
   | `05_pure_python/` | Double integers | `apply()` only, no LLM |
   | `06_stream_big/` | Square 1 M numbers with StreamWriter | Streaming + performance |
   | `07_multistep_rag/` | Simple Retrieval-Augmented QA | Multi-step, history templating |
   | `08_chatbot_yaml/` | YAML-declared customer support bot | Full YAML pipeline |
   | `09_clinical_qa/` | Symptoms → diagnosis suggestion | Domain demo, 3 steps |
   | `10_eval_loop/` | Automatic metric eval of generated JSON | Metrics step & batch processing |
   | `11_tool_use/` | LLM selects and calls a mock tool (math or weather) | Tool calling pattern |
   | `12_flagship/` | End-to-end pipeline combining retrieval, translation, summarisation, and evaluation | All-star demo |

   Each directory contains `chain.py` (or `.yml`), `inputs.jsonl`, and a README with expected DAG & sample output.

   Legacy prototypes were moved to `_legacy_examples/`.

-----------------------------------------------------------------------------
## 3 – Migration Phases & Checklist

We iterate per the *tick-box* workflow – implement → test → tick → commit.

- [ ] **1. Branch created `code-cleanup`** *(automated)*
- [ ] **2. Audit utils/ folder** – list candidates for deletion/merge.
- [ ] **3. Remove dead logging versions** – confirm `utils.logging_v3` superset.
- [ ] **4. Rename `logging_v3` to `logging` & update imports**.
- [ ] **5. Purge comments / experimental TODOs** across library (grep for `TODO`, `HACK`, `DEBUG`).
- [ ] **6. Delete redundant example files** – keep a temporary backup in `_legacy_examples/` for diff reference.
- [x] **7. Scaffold new `examples/` tree with placeholders + READMEs**.
   ```bash
   mkdir -p examples/{basic,branching,pure_python,streaming,cli_yaml}
   # created placeholder chain.py, inputs.jsonl, README.md for each
   ```
- [x] **8. Implement `01_sentiment` example & verify.**
   ```bash
   poetry run chainette run examples/01_sentiment/chain.py sentiment_chain \
       examples/01_sentiment/inputs.jsonl _tmp_sentiment_out --quiet --no-icons
   # outputs OK ➜ _tmp_sentiment_out/flattened/0.jsonl
   ```
- [x] **9. Implement `02_fin_metrics` example.**
   ```bash
   poetry run chainette run examples/02_fin_metrics/chain.py fin_metrics_chain \
       examples/02_fin_metrics/inputs.jsonl _out_fin --quiet --no-icons
   ```
- [x] **10. Implement `03_translate_summary` example.**
   ```bash
   poetry run chainette run examples/03_translate_summary/chain.py translate_chain \
       examples/03_translate_summary/inputs.jsonl _out_trans --quiet --no-icons
   ```
- [ ] **11. Implement `04_branch_math` example.**
- [ ] **12. Implement `10_eval_loop` example.**
- [ ] **13. Implement `11_tool_use` example (tool selection).**
- [ ] **14. Implement `12_flagship` example (comprehensive showcase).**
- [ ] **15. Update root README with catalogue & screenshot.**
- [ ] **16. Ensure all tests green (`poetry run pytest -q`).**
- [ ] **17. Bump version & update CHANGELOG.**