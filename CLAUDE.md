# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Undergraduate thesis implementing the **Reflexion** framework (NeurIPS 2023) in Python to mitigate cognitive bias in LLM responses to Q&A questions about a Brazilian PIX financial dataset. The dataset is not yet included — all dataset-touching components use abstract interfaces to be filled in later.

## Commands

```bash
# Install dependencies
uv add <package>
uv add --dev <package>

# Run a single Reflexion trial
uv run python scripts/run_reflexion.py \
  --question "Qual foi o volume total de transações PIX em 2023?" \
  --ground-truth "1.2 trilhão de reais"

# Run tests
uv run pytest

# Custom config
uv run python scripts/run_reflexion.py --config config/default.yaml --question "..." --ground-truth "..."
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

## Architecture

The system implements **Algorithm 1** from the Reflexion paper — a loop of three LLM-powered components:

1. **Actor** (`src/reflexion/actor.py`) — answers the question. Two strategies:
   - `CoTActor`: single LLM call with chain-of-thought prompting
   - `ReActActor`: multi-step Think→Action→Observation loop using a `ToolRegistry`
2. **Evaluator** (`src/reflexion/evaluator.py`) — scores the answer (pass/fail). Two variants:
   - `ExactMatchEvaluator`: normalised string comparison
   - `LLMJudgeEvaluator`: LLM call, parses "PASS/FAIL: reason"
3. **SelfReflection** (`src/reflexion/reflection.py`) — given the failed trajectory + eval feedback, generates a verbal critique stored in memory for the next trial

**Orchestration:** `ReflexionAgent` in `src/reflexion/agent.py` runs the loop and accumulates `TokenUsage` (input/cached/output tokens + cost in USD) across all trials.

**LLM layer** (`src/reflexion/llm/`) supports Anthropic and OpenAI, each component independently configurable in `config/default.yaml`. Cost is calculated from a pricing table in `src/reflexion/llm/base.py`.

**Prompts** live in `prompts/prompts.yaml` as structured YAML with `system`/`user` keys. `PromptLoader` renders them with `string.Template` (`$variable` syntax).

## Extending for the PIX Dataset

When the dataset arrives:
1. Add concrete tools in `src/reflexion/tools.py` (subclass `Tool`, register in `ToolRegistry`)
2. Add `PIXDatasetLoader` in `datasets/`
3. Optionally add `BiasClassifierEvaluator` in `src/reflexion/evaluator.py`
4. Add a batch runner `scripts/run_batch.py` for dataset-scale evaluation
