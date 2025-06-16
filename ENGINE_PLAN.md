# Chainette Engine Elegance Plan  
> Branch: `engine-refactor` (branched from `runner`)

This blueprint fixes lazy-load / release logic and removes the redundant  
`input_model` field—while staying tiny and readable per **llm.txt**.

---

## 1 – Problems Today (June 2025)

| Area          | Issue                                                                                                      |
|---------------|------------------------------------------------------------------------------------------------------------|
| Lazy spin-up  | `Step` spins engines, `Executor` sometimes releases ⇒ double-duty, bugs on branch model-switch.            |
| Release       | `release_engine()` sprinkled around; leaks GPU mem on failures.                                            |
| Pooling       | `pool.py` LRU lacks ref-count; branch can kill engine still in use.                                        |
| Schema        | Every node asks for `input_model`, but full history is available ⇢ field is noise.                         |

---

## 2 – Target Design (≤ 300 LOC)

```mermaid
flowchart TD
    subgraph EngineLayer
        Step -->|request| EngineBroker
        Branch -->|request| EngineBroker
        EngineBroker --> Pool
        Pool -->|ctx mgr| LiveEngine[vLLM / Ollama]
    end
```

### Components

1. **`engine/broker.py`** (≤ 80 LOC)  
   * `with acquire(name) as eng:` – returns live engine, bumps `ref_count`.  
   * `flush(force=False)` – releases idle engines or all when `force=True`.

2. **`engine/engine_pool.py`** (≤ 70 LOC)  
   * Dict `name → LiveEngineWrapper(engine, ref_count, last_used_ts)`.  
   * Wrapper's `__exit__` decrements count; 0 ⇒ eligible for flush.

3. **Policy constants** (≤ 20 LOC)  
   * `IDLE_SEC = 180` etc.

4. **Step changes** (≤ 30 LOC diff)

   ```python
   from chainette.engine.broker import EngineBroker

   with EngineBroker.acquire(self.engine_name) as eng:
       raw = eng.generate(prompts, self.sampling)
   ```

5. **Executor cleanup** (≤ 20 LOC diff)  
   * Remove manual engine-switch block.  
   * Call `EngineBroker.flush(force=True)` at run end.

6. **Drop `input_model`** (≤ 60 LOC total edits)  
   * Delete arg from `Step` / `ApplyNode`; examples & CLI updated.  
   * CLI: if first node is Apply with no declared model, parse inputs as `dict` and warn.

---

## 3 – Roadmap / TODO

- [x] **Design docs** – sequence diagram + LOC budget table (see `ENGINE_DESIGN.md`).
- Engine layer  
  - [ ] `broker.py`