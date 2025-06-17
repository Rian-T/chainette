# Engine Broker / Pool – Design Doc

> Companion to `ENGINE_PLAN.md`

This doc provides a concise **sequence diagram** of request/flush flow and a quick **LOC budget** reference.

## Sequence – acquire / release / flush

```mermaid
sequenceDiagram
    participant Step
    participant Broker
    participant Pool
    participant LiveEngine

    Step->>Broker: acquire("gemma_ollama")
    Broker->>Pool: get(name)
    alt engine alive
        Pool-->>Broker: wrapper (ref++)
    else first time
        Pool->>LiveEngine: spin-up vLLM/Ollama
        Pool-->>Broker: wrapper (ref=1)
    end
    Broker-->>Step: context mgr (LiveEngine)
    Step->>LiveEngine: generate(prompts)
    Step-->>Broker: __exit__
    Broker->>Pool: dec ref
    alt ref==0 & idle>IDLE_SEC
        Pool->>LiveEngine: release_engine()
    end

    Note over Broker,Pool: Executor calls Broker.flush(force=True)
    Broker->>Pool: flush all
    Pool->>LiveEngine: release_engine()
```

## File Layout & LOC budget

| File | Purpose | Max LOC |
|------|---------|---------|
| `engine/broker.py` | acquire / flush wrappers | **≤ 80** |
| `engine/engine_pool.py` | ref-count, idle timing | **≤ 70** |
| `core/step.py` diff | replace `self.engine` usage | **+30** |
| `core/executor.py` diff | remove engine switch, add flush | **–40** |

Total new code ≤ **140 LOC**, net core diff negative.

---

*Last updated: 2025-06-16* 