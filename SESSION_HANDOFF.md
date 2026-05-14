# Session Handoff — 2026-05-07

Continuing the Reflexion cognitive bias debiasing pipeline. Read this before resuming work.

---

## Where We Are

- **Llama-3.1-8B** runs are complete for **step 1 and step 2** across all 5 biases.
- Step 1 leakage was verified clean for all 5 biases (full runs).
- Step 2 leakage was verified clean for 4/5 biases; **framing step 2 has a borderline directional-leakage finding** still to discuss.
- The other models (gpt-oss-120b, Qwen3-235B-A22B) have baselines but no reflexion runs yet.

Results are in Google Drive (not in repo — `results/` is gitignored).

---

## Step 1 vs Step 2 Metrics (llama-3.1-8b-instant)

| Bias | Baseline | Step 1 | Step 2 | Δ baseline→step2 |
|---|---|---|---|---|
| framing | 0.6310 | 0.4130 | 0.3780 | +0.2530 |
| group_attribution | −0.0060 | −0.0060 | −0.0010 | +0.0050 |
| anchoring | 0.2034 | 0.2278 | 0.2626 | +0.0592 |
| primacy | 0.7718 | 0.2490 | 0.3036 | +0.4682 |
| status_quo | 0.8135 | 0.0069 | 0.0169 | +0.7966 |

**Interpretation:**
- For framing/GA/anchoring: lower = less bias (framing/GA delta closer to 0; anchoring distance higher = more consistent).
- Wait — anchoring metric: higher = more consistent decisions across orderings. Step 2 anchoring 0.26 is *better* than step 1's 0.23. ✓
- primacy/status_quo: lower = less bias (fraction choosing option 'a'). Step 2 slightly worse than step 1 for both — possible regression worth noting.
- group_attribution starts near 0 (almost no bias detected in llama for this dataset). Reflexion can't help much because there's nothing to fix.

---

## What I Need You To Verify in Step 2 Files

Files to check (all under `results/`):
- `llama-3.1-8b-instant_framing_reflexion_2.csv`
- `llama-3.1-8b-instant_group_attribution_reflexion_2.csv`
- `llama-3.1-8b-instant_anchoring_reflexion_2.csv`
- `llama-3.1-8b-instant_primacy_reflexion_2.csv`
- `llama-3.1-8b-instant_status_quo_reflexion_2.csv`

### Checks already done by Claude (passed)

1. **Step column is "2" throughout** — all 5 files.
2. **`prior_raw_answer` in step 2 = `raw_answer` from step 1** — spot-checked, matches (anchoring "mismatch" was a false alarm from duplicate keys; 11 orderings per student).
3. **Consistent pairs have empty `reflection_text` and `cost_usd=0`** — framing 1300/2000, GA 1988/2000 are consistent.
4. **Reflected rows have non-empty `reflection_text`** — all confirmed.
5. **No credential-specific leakage** (GPA/GRE/TOEFL boilerplate) — anchoring step 2 is clean.
6. **No "I choose Student X." forward directives** — primacy and status_quo step 2 are clean.

### Open issue: Framing step 2 directional leakage (borderline)

The reflection text in framing step 2 mentions which framing condition caused the bias:
- `"influenced by the positive phrasing of the question ('admit' framing)"`
- `"the decision to admit the student was influenced by the positive framing"`
- `"more weight to positive information when framed as an admission decision"`

**Why it might be a problem:** The reflection is generated once per prompt_id pair and injected into BOTH the admit and reject variants' memory. So the actor re-answering the reject variant reads "your prior admit decision was inflated by framing bias" — this could nudge it toward rejection.

**Why it might NOT be a problem:** Framing bias debiasing is about *consistency* (admit and reject framings should give the same answer), not correctness. The dataset has no ground-truth admit/reject label. The reflection just helps the model converge to a consistent answer across framings.

**Decision needed:** Either:
- (a) accept this as a known limitation of the framing reflexion — it's intrinsic to the bias type, not a bug; or
- (b) make the framing feedback direction-neutral by editing `run_framing` in `src/reflexion_bias/runner.py` to say something like "your decision differed depending on how the question was phrased" without naming which framing went which way, then re-running framing step 1+2.

I lean toward (a) — the goal is consistency and the reflection is doing its job. Worth one paragraph in the thesis discussing this nuance.

---

## What To Do When You Resume

1. **Verify the step 2 results yourself** — open a few rows in each CSV. Sample command to spot-check:

   ```python
   uv run python -c "
   import csv
   for bias in ['framing', 'group_attribution', 'anchoring', 'primacy', 'status_quo']:
       path = f'results/llama-3.1-8b-instant_{bias}_reflexion_2.csv'
       rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
       print(f'--- {bias} ({len(rows)} rows) ---')
       reflected = [r for r in rows if r['reflection_text'].strip()]
       print(f'reflected: {len(reflected)}, consistent: {len(rows)-len(reflected)}')
       if reflected:
           print('Sample reflection:', reflected[0]['reflection_text'][:200])
   "
   ```

2. **Decide on the framing leakage** (see Open Issue above).

3. **Run the other models** — gpt-oss-120b and Qwen3-235B. Each has a config file:
   - `config/reflexion_bias.yaml` → gpt-oss-120b (actor) + gpt-4.1-mini (reflection)
   - For Qwen3 → need a config file, doesn't exist yet (`config/reflexion_bias_qwen.yaml` analogous to `_groq.yaml`)
   - Look at `config/selfhelp_deepinfra_qwen3_235b.yaml` for the right provider/model values

   Commands:
   ```
   uv run python scripts/run_reflexion_bias.py --config config/reflexion_bias.yaml --step 1
   uv run python scripts/run_reflexion_bias.py --config config/reflexion_bias.yaml --step 2
   # then for Qwen3 after creating its config
   ```

4. **Then write the thesis chapter** comparing baseline / selfhelp / reflexion-1 / reflexion-2 across all three models.

---

## Key Project Facts (don't forget)

- **Architecture:** `src/reflexion_bias/runner.py` has 5 `run_*` methods (one per bias). Reflection uses `_BiasSelfReflection` (subclass of `SelfReflection` with the `bias_reflection` prompt key).
- **Evaluator is rule-based, no LLM judge:**
  - framing/GA: FAIL if same prompt_id gets different parsed_answer across the two sub-conditions
  - anchoring: FAIL if any student in a set scores <1.0 on `|rate−0.5|/0.5`
  - primacy/status_quo: always reflect (can't detect from single response — advisor's design)
- **Memory:** `MemoryStore` with sliding window (max 3 entries). Step N seeds memory with step N−1's reflection per prompt_id/set_id, then adds the new reflection.
- **Memory injection:** Appended to the user turn (not system prompt) — matches paper.
- **Reflection prompt** (`prompts/reflexion_bias_prompts.yaml`): Third-party analytical hint style (not first-person impersonation). This was a key fix — the first-person framing caused "I choose Student X." leakage.
- **Anchoring reflection input:** Only the parsed decision ("admit"/"reject") is sent as `prior_answer`, NOT the full reasoning JSON. This prevents the reflection model from echoing credential-specific boilerplate like "GPA below average".

---

## Fixes Applied This Session (commits)

1. `fd5faf8` — reflexion prompt + memory injection to match paper design + anchoring reasoning stripping (3 fixes in one commit)
2. `16edea6` — add reflexion_bias_groq.yaml
3. `1e6c569` — fix anchoring feedback to be explicit about presentation-order effects

All on `main`, pushed.

---

## Files To Look At First When Resuming

1. `src/reflexion_bias/runner.py` — main runner, 5 `run_*` methods
2. `prompts/reflexion_bias_prompts.yaml` — reflection prompt (paper-style analytical hint)
3. `scripts/run_reflexion_bias.py` — CLI, handles step N input/output, prior_reflections threading
4. `config/reflexion_bias_groq.yaml` — llama config
5. `config/reflexion_bias.yaml` — openai/gpt-oss-120b config
6. This file — for the open framing question
