# Chainette 🪢

**Chainette** is a tiny, type‑safe way to compose LLM pipelines in Python.

*   ⚖️ < 350 LOC • MIT
*   🔌 Works with any vLLM‑served model (Llama 3, Gemma, Mixtral…)
*   📜 Inputs & outputs are **Pydantic** models – no more brittle string parsing
*   🎯 Automatic JSON **guided decoding**: the model must reply with the schema you declare
*   🗂️ Filesystem first – every run leaves reproducible artefacts (`graph.json`, step files, flattened view)
*   🖥️ Simple CLI: `warmup | run | kill | inspect`

## Install

```bash
pip install chainette           # or: poetry add chainette
```

## Quick Start

Define your models, register an engine, create steps, and build a chain:

```python
from pydantic import BaseModel
from chainette import Step, Chain, SamplingParams, register_engine

# 1. Register an engine
register_engine(
    name="llama3",
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.6,
    lazy=True,  # only start when needed
)

# 2. Define input/output schemas
class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    confidence: float

# 3. Create a step
qa_step = Step(
    id="qa",
    name="Question Answering",
    input_model=Question,
    output_model=Answer,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="Answer questions accurately and concisely.",
    user_prompt="Question: {{text}}",
)

# 4. Create a chain
chain = Chain(name="Simple QA", steps=[qa_step])

# 5. Run the chain
results = chain.run([
    Question(text="What is the capital of France?"),
    Question(text="How do transformers work?")
])
```

## Core Concepts

### Steps

A `Step` is a single LLM task with defined input and output models. Each step:
- Uses **guided JSON decoding** to ensure output follows your schema
- Handles batching for efficient processing
- Tracks ID, name, and other metadata for the run

### Chains

A `Chain` executes a sequence of steps, handling:
- Data flow between steps
- Parallelism with branches
- Batching of inputs
- Output serialization
- Execution metadata

### Branches

Use `Branch` for parallel workflows:

```python
chain = Chain(
    name="Translation Chain",
    steps=[
        extract_step,
        [  # Parallel branches
            Branch(name="fr", steps=[translate_french]),
            Branch(name="es", steps=[translate_spanish]),
        ],
    ],
)
```

### Branch Joins

Merge outputs from parallel branches back into the main flow with `Branch.join(alias)`. The final output of each branch becomes accessible in later templates via the alias you provide:

```python
fr_branch = Branch(name="fr", steps=[translate_french]).join("fr")
es_branch = Branch(name="es", steps=[translate_spanish]).join("es")

agg = Step(
    id="agg",
    input_model=Translation,
    output_model=Summary,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0),
    system_prompt="Summarise both translations.",
    user_prompt="FR: {{fr.translated}}\nES: {{es.translated}}",
)

chain = Chain(name="Translate & Summarise", steps=[[fr_branch, es_branch], agg])
```

### Python Functions

Inject pure Python functions with `apply`:

```python
from chainette import apply

def filter_low_confidence(items):
    return [item for item in items if item.confidence > 0.7]

chain = Chain(
    name="QA with filtering",
    steps=[qa_step, apply(filter_low_confidence)],
)
```

### Architecture Overview (v2 – Elegance refactor)

Internally Chainette is now modelled as a **directed acyclic graph**:

```
inputs → Step/Apply → … → Branch(es) → outputs
```

Key runtime components:

• `Graph` / `GraphNode` – tiny DATACLASS helpers to link nodes.<br/>
• `Executor` – walks the graph depth-first, handles batching & engine reuse.<br/>
• `AsyncExecutor` – same but with an `async def run` using `anyio` threads.<br/>
• `EnginePool` – LRU cache of live vLLM / Ollama engines.<br/>
• `Result` – wrapper object to propagate either a `value` **or** an `error`.

You still build chains exactly the same way; `Chain.run()` now proxies to the
executor under the hood, so existing code doesn't change.

## Runner Improvements (June 2025)

Chainette now ships with a streaming execution **Runner**:

* Chunked batching via `Executor.run_iter` → constant-memory even on millions of rows.
* `StreamWriter` flushes each batch immediately and rolls files (`000.jsonl`, `001.jsonl`, …). Optional Parquet support (`pyarrow`).
* Rich live logger—ASCII banner, DAG tree, progress bars—powered by an EventBus.
* CLI additions:
  * `--stream-writer` flag (recommended for big runs).
  * `--quiet / --json-logs` for headless or scripted runs.
  * `inspect-dag` command to visualise the graph without execution.

Quick demo:
```bash
poetry run chainette run examples/runner/huge_batch_demo.py demo_chain inputs_huge.jsonl out --stream-writer
```

## CLI Usage

Chainette provides a simple CLI for managing LLM engines:

```bash
# Start non-lazy engines
chainette warmup -f engines.yml -e llama3

# Run a chain from a module
chainette run examples.qa:my_chain -i inputs.json --output_dir results

# Run a chain using huggingface datasets
chainette run examples.qa:my_chain -i dataset_name/split_name --columns input_column_name_1,input_column_name_2 --output_dir results

# Terminate engines
chainette kill --all
```

## Engine Configuration

Configure engines in YAML:

```yaml
engines:
  - name: llama3
    model: NousResearch/Meta-Llama-3-8B-Instruct
    dtype: bfloat16
    gpu_memory_utilization: 0.6
    enable_reasoning: false
    devices: [0]
    lazy: true
```

## Output Structure

Each run creates:
- `graph.json`: Chain execution metadata
- Step output directories with data in JSON/CSV
- Optional flattened view combining all steps

## Examples

Check the `examples/` directory:
- `product_struct_extract.py`: Extract product attributes with translations
- `multi_doc_summary_eval.py`: Document summarization with quality scoring
- `join/inc_dec_join.py`: Tiny pure-Python Join demo (no LLM needed)

## Requirements

- Python 3.9+
- vLLM for model serving
- Pydantic v2
- Hugging Face datasets

## License

MIT

## Engine Broker (2025)

A minimal ref-count abstraction ensuring engines spin up lazily and are flushed deterministically.

```python
from chainette.engine.broker import EngineBroker as EB

with EB.acquire("gemma_ollama") as eng:
    eng.generate(prompts, sampling)

# At end of run Executor calls
EB.flush(force=True)  # frees any idle engines
```

Key points
1. Context-manager → zero manual release in nodes.
2. Reference counting prevents premature kills while branches share an engine.
3. Idle engines auto-evict after 180 s or via `force=True`.
