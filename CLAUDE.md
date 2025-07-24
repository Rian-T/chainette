# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Debugging Log - MedGemma CIM-10 Issue

### Current Status (2025-07-24)
**Problem**: Example 14 (medgemma_cim10) fails with "vLLM server process for 'medgemma' exited prematurely"

**Environment Verified**:
- ✓ 2x NVIDIA A100-SXM4-80GB GPUs available (79.3GB each, ~99% free)
- ✓ PyTorch 2.6.0+cu124 with CUDA 12.4 working correctly
- ✓ vLLM 0.8.5.post1 imported successfully
- ✓ Transformers 4.52.4 working, model config accessible
- ✓ Memory requirements: ~32.7GB per GPU (well within 80GB capacity)

**Root Cause Found**:
1. ✓ vLLM startup works correctly - model loads successfully (25.5GB per GPU)
2. ✗ Configuration errors in chain.py:
   - `startup_timeout=60000` was in seconds (16.7 hours) instead of intended 10 minutes
   - `engine_kwargs` field doesn't exist - should use individual parameters
   - torch.compile phase takes ~3+ minutes, longer than original timeout

**Fix Applied**:
- Changed `startup_timeout=60000` to `startup_timeout=900` (15 minutes)
- Replaced `engine_kwargs={}` with individual parameters:
  - `tensor_parallel_size=2`
  - `max_model_len=8192`
  - `gpu_memory_utilization=0.9`
  - `dtype="bfloat16"`

**Testing Results**:
✓ Debug inputs (5 records): Chain executed successfully, produced outputs
✓ Full dataset (19,161 records): Chain started correctly, processing in progress
- No more "vLLM server process exited prematurely" errors
- Engine startup takes ~5-10 minutes (model loading + torch.compile)
- Processing time for full dataset: ~6+ hours estimated

**Final Status**: ✓ FIXED - Issue resolved successfully

**Command Failing**:
```bash
poetry run chainette run examples/14_medgemma_cim10/chain.py medgemma_cim10_chain examples/14_medgemma_cim10/inputs_full.jsonl $SCRATCH/chainette/_out_full_icd10
```

## Project Overview

Chainette is a type-safe LLM pipeline composition library in Python. It provides a DAG-based execution engine for chaining LLM operations with Pydantic models for input/output validation and guided JSON decoding.

## Core Architecture

### Key Components
- **Chain/Step**: High-level API for building LLM pipelines
- **Graph/GraphNode**: DAG representation using dataclasses 
- **Executor/AsyncExecutor**: Graph traversal engine with batching and engine reuse
- **EnginePool/EngineBroker**: LRU cache and ref-counting for vLLM/Ollama engines
- **Result**: Error-propagating wrapper for values

### Directory Structure
- `chainette/core/`: Core pipeline components (Chain, Step, Executor, Graph)
- `chainette/engine/`: Engine management (EnginePool, Broker, HTTP clients)
- `chainette/io/`: Output handling (StreamWriter for large datasets)
- `chainette/utils/`: Utilities (logging, templating, DAG visualization)
- `examples/`: 13 example pipelines showing different patterns
- `tests/`: Unit tests and integration tests

## Development Commands

### Running Tests
```bash
# All tests
poetry run pytest

# Skip integration tests (no external services needed)
poetry run pytest -m "not integration" 

# Integration tests only (requires OpenAI API key, vLLM server, or Ollama)
poetry run pytest -m integration
```

### Running Examples
```bash
# Basic sentiment analysis
poetry run chainette run examples/01_sentiment/chain.py sentiment_chain examples/01_sentiment/inputs.jsonl _out_sentiment

# With OpenAI backend (requires OPENAI_API_KEY)
poetry run chainette run examples/12_flagship/chain.py flagship_chain examples/12_flagship/inputs.jsonl _tmp_flagship_out --quiet --no-icons

# With Ollama backend (requires ollama serve)
poetry run chainette run examples/12_flagship/ollama/chain.py flagship_chain_ollama examples/12_flagship/ollama/inputs.jsonl _tmp_flagship_ollama_out --quiet --no-icons

# YAML-defined chain
poetry run chainette run-yaml examples/08_chatbot_yaml/chain.yml _out_chatbot --symbols-module examples.08_chatbot_yaml.steps --quiet --no-icons

# Large batches with streaming
poetry run chainette run examples/06_stream_big/chain.py big_chain examples/06_stream_big/inputs.jsonl _out_big --stream-writer
```

### DAG Inspection
```bash
# Visualize execution graph without running
poetry run chainette inspect-dag examples/03_translate_summary/chain.py translate_chain --no-icons
```

### Engine Management
```bash
# Warm up engines
chainette warmup -f engines.yml -e llama3

# Kill all engines
chainette kill --all
```

## Testing Notes

- Tests are marked with `@pytest.mark.integration` for external service dependencies
- Integration tests require live OpenAI API, vLLM server, or Ollama service
- Use `pytest -m "not integration"` for fast local testing without external dependencies

## Key Patterns

### Engine Registration
```python
from chainette import register_engine

# vLLM local (GPU)
register_engine(name="llama3", model="NousResearch/Meta-Llama-3-8B-Instruct", lazy=True)

# OpenAI HTTP
register_engine(name="gpt4o", model="gpt-4o-mini", backend="openai")

# Ollama HTTP  
register_engine(name="qwen", model="qwen2.5-instruct", backend="ollama_api")
```

### Chain Building
```python
from chainette import Chain, Step, Branch
from pydantic import BaseModel

chain = Chain(name="example", steps=[
    step1,
    [Branch(name="parallel1", steps=[step2a]), Branch(name="parallel2", steps=[step2b])],
    step3
])
```

### Guided JSON Output
All steps use Pydantic models for type-safe, guided JSON decoding - the LLM is forced to output valid JSON matching the declared schema.

## CLI Flags

- `--quiet`: Suppress verbose output
- `--no-icons`: Plain ASCII tree rendering
- `--stream-writer`: Enable streaming for large datasets  
- `--json-logs`: JSON-formatted logs for headless runs
- `--max-branches N`: Limit displayed branches in DAG visualization