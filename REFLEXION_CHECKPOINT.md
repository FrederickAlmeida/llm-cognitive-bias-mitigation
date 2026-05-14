# Reflexion Implementation Checkpoint

**Last updated:** 2026-03-24
**Status:** Skeleton complete, awaiting PIX dataset
**Latest commit:** `fe56677` — "Implement Reflexion framework skeleton for cognitive bias mitigation"

---

## What's Complete

### Core Framework
- ✅ **LLM Layer** (`src/reflexion/llm/`)
  - Anthropic client with token usage + cost calculation
  - OpenAI client with token usage + cost calculation
  - Pricing table for all models (input, cached input, output tokens)
  - `LLMClient` abstract base + `make_client()` factory

- ✅ **Memory** (`src/reflexion/memory.py`)
  - Bounded sliding window (default 3 entries)
  - `MemoryStore` with add/clear/format methods

- ✅ **Tools** (`src/reflexion/tools.py`)
  - Abstract `Tool` interface (name, description, run method)
  - `ToolRegistry` for register/get/descriptions
  - `FinishTool` auto-registered (signals end of ReAct loop)

- ✅ **Actor** (`src/reflexion/actor.py`)
  - `CoTActor` — single LLM call with answer extraction
  - `ReActActor` — think/act/observe multi-step loop
  - ReAct parsing: `Action: ToolName[input]` format
  - Parse error recovery (injects corrective observation)
  - `make_actor()` factory

- ✅ **Evaluator** (`src/reflexion/evaluator.py`)
  - `ExactMatchEvaluator` — normalised string comparison
  - `LLMJudgeEvaluator` — LLM call parsing "PASS/FAIL: reason"
  - `make_evaluator()` factory
  - `EvalResult` with feedback + optional token usage

- ✅ **Self-Reflection** (`src/reflexion/reflection.py`)
  - Takes failed trajectory + evaluator feedback
  - Generates 2–4 sentence verbal critique
  - Returns (text, LLMResponse) tuple for cost tracking

- ✅ **Agent Loop** (`src/reflexion/agent.py`)
  - `ReflexionAgent` implements Algorithm 1 from paper
  - Orchestrates: Actor → Evaluator → SelfReflection → Memory
  - Accumulates token usage + cost across all trials
  - `ReflexionResult` with full metadata
  - `PromptLoader` loads YAML and renders with `$variable` substitution

### Configuration & Prompts
- ✅ `config/default.yaml` — all settings (actor strategy, LLM providers/models, max trials, memory size)
- ✅ `prompts/prompts.yaml` — all prompt templates (actor_cot, actor_react, self_reflection, evaluator_judge)
- ✅ `.env.example` — API key template
- ✅ `CLAUDE.md` — developer guidance

### CLI & Entry Point
- ✅ `scripts/run_reflexion.py` — CLI with argparse
  - Takes `--question`, `--ground-truth`, `--config`
  - Builds agent from config
  - Outputs summary with solved/trials/answer/reflections/cost

### Testing
- ✅ 74 unit tests (all passing)
- ✅ No API key dependencies (all mocked)
- ✅ Test coverage:
  - Token cost calculation
  - Memory sliding window
  - Tool registry
  - Actor strategies (CoT, ReAct, parse recovery)
  - Evaluators (exact match, LLM judge)
  - Self-reflection prompt injection
  - Full agent loop (1st trial win, reflection → win, max trials exhausted)
  - Usage/cost accumulation

---

## What's Pending

### 1. Dataset Integration (HIGH PRIORITY)
**When the PIX dataset arrives:**

- [ ] Create `datasets/pix_loader.py`
  - `PIXDatasetLoader` class
  - Load from CSV/JSON
  - Return list of (question, ground_truth) tuples

- [ ] Extend `src/reflexion/tools.py`
  - Add `PIXLookupTool` — search dataset by key/ID
  - Add `PIXAggregateTool` — sum/count/average operations
  - Register tools in `ToolRegistry` when dataset loads

### 2. Bias Detection & Classification (MEDIUM PRIORITY)
**After dataset tools work:**

- [ ] Add `BiasClassifierEvaluator` in `src/reflexion/evaluator.py`
  - LLM call that classifies which cognitive bias caused the failure
  - Requires: labeled bias categories (anchoring, availability heuristic, framing, confirmation bias, etc.)
  - Integrate into self-reflection prompt

- [ ] Enhance self-reflection prompt in `prompts/prompts.yaml`
  - Add explicit instruction to identify bias type
  - Example: "Identify the cognitive bias (anchoring, availability, framing, confirmation, or other)"

### 3. Batch Evaluation (MEDIUM PRIORITY)
**After dataset + evaluators work:**

- [ ] Create `scripts/run_batch.py`
  - Load PIX dataset via `PIXDatasetLoader`
  - Run Reflexion on each question
  - Log results to CSV/JSON with columns:
    - `question`, `ground_truth`, `final_answer`, `solved`, `trials_used`
    - `total_cost_usd`, `total_input_tokens`, `total_output_tokens`
    - `detected_bias` (if classifier available)
    - `reflections` (semicolon-separated or JSON)

### 4. Ablation Experiments (LOW PRIORITY, POST-DATASET)
**Test different configurations:**

- [ ] Create `config/ablations/` directory with variants:
  - `cot_exact_match.yaml` — CoT + exact match evaluator
  - `react_llm_judge.yaml` — ReAct + LLM judge evaluator
  - etc.

- [ ] Test variables:
  - Actor strategy: CoT vs ReAct
  - Evaluator: exact match vs LLM judge
  - Max trials: 1, 3, 5
  - Memory size: 1, 3, 5

---

## How to Resume

### Verify Current State
```bash
# Check latest commit
git log --oneline | head -3

# Run tests (verify nothing broke)
uv run pytest tests/ -v

# Quick CLI test (requires .env with API keys)
uv run python scripts/run_reflexion.py \
  --question "Qual é a chave PIX mais comum?" \
  --ground-truth "CPF"
```

### Next Steps (in order)
1. **Get the PIX dataset** → create `datasets/pix_loader.py`
2. **Register dataset tools** → extend `src/reflexion/tools.py`
3. **Test ReAct with tools** → verify agent can use lookup/aggregate
4. **Add bias classifier** → create `BiasClassifierEvaluator`
5. **Build batch runner** → `scripts/run_batch.py`
6. **Run ablations** → test different configs on full dataset

---

## Key Design Decisions

### Why No LangChain/LangGraph?
The Reflexion loop is simple (3 function calls in a `for` loop). A framework would add abstraction that obscures the implementation — important for a thesis where you need to explain exactly what the algorithm does.

### Why Config + YAML Prompts?
- Config allows quick ablation experiments without code changes
- YAML prompts are human-readable and easy to refine based on dataset feedback
- `$variable` substitution is simple and doesn't require a template engine

### Why Independent LLM Providers per Component?
Each of Actor, Evaluator, SelfReflection can use different models. This enables ablations like "Claude for actor + GPT-4 for evaluator" and cost optimization.

### Why Token Cost Tracking?
For a thesis on a financial dataset, showing cost efficiency matters. Every LLM call is tracked with input/cached/output tokens and USD cost.

---

## Project Structure
```
src/reflexion/
  llm/
    base.py              # TokenUsage, LLMResponse, PRICING, calculate_cost, LLMClient (ABC)
    anthropic_client.py  # AnthropicClient implementation
    openai_client.py     # OpenAIClient implementation
    __init__.py          # make_client factory
  actor.py               # CoTActor, ReActActor, Trajectory, make_actor factory
  agent.py               # ReflexionAgent, ReflexionResult, PromptLoader
  evaluator.py           # ExactMatchEvaluator, LLMJudgeEvaluator, EvalResult, make_evaluator factory
  reflection.py          # SelfReflection
  memory.py              # MemoryStore
  tools.py               # Tool (ABC), ToolRegistry, FinishTool, ToolResult
  __init__.py            # Public API exports

config/
  default.yaml           # Runtime config (all settings)

prompts/
  prompts.yaml           # All prompt templates (system/user pairs)

scripts/
  run_reflexion.py       # CLI entry point

tests/                   # 74 unit tests (not committed, in .gitignore)
```

---

## Testing Checklist (when resuming)
- [ ] Unit tests pass: `uv run pytest tests/ -v`
- [ ] CLI works: `uv run python scripts/run_reflexion.py --question "..." --ground-truth "..."`
- [ ] Can switch actor strategy (CoT vs ReAct) via config
- [ ] Can switch evaluator type (exact_match vs llm_judge) via config
- [ ] Token usage accumulates correctly across trials
- [ ] Cost calculation matches model pricing table
- [ ] Memory fills and bounds correctly (max 3 entries)
