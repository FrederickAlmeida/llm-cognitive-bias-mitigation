# Self-Help Debiasing Results: Llama 3.1 8B Instant

**Model:** Groq `llama-3.1-8b-instant`  
**Framework:** Self-Help (Echterhoff et al., EMNLP 2024)  
**Date:** April 1, 2026

## Summary

Evaluated the effectiveness of the Self-Help debiasing technique on llama-3.1-8b-instant across all five cognitive biases from the BiasBuster dataset. Self-Help prompts the model to rewrite biased prompts (or decisions) to remove cognitive bias cues, then measures if the rewritten prompts lead to less biased answers.

## Results

| Bias | Baseline | Self-Help | Improvement | n | Status |
|---|---|---|---|---|---|
| **Framing** | 0.2560 | -0.1900 | +0.0660 | 2000 | ⚠️ Flipped |
| **Group Attribution** | -0.0180 | 0.1250 | **-0.1070** | 2000 | ❌ Worsened |
| **Status Quo** | 0.0635 | 0.2401 | **-0.1766** | 1008 | ❌ Worsened |
| **Primacy** | 0.7520 | 0.5060 | **+0.2460** | 1008 | ✅ Improved |
| **Anchoring** | 0.2153 | 0.1433 | **-0.0721** | 5449 | ❌ Worsened |

## Interpretation

### Metric Definitions
- **For most biases** (framing, group_attribution, status_quo, primacy): Lower absolute value = less bias. Positive improvement = reduction in bias magnitude.
- **For anchoring**: Higher value = more consistent (less biased) decisions across orderings. Positive improvement = higher consistency.

### Key Findings

#### ✅ Primacy (+0.2460)
Self-Help successfully reduced primacy bias by ~33%. The model became less influenced by option position when given rewritten prompts.

#### ⚠️ Framing (+0.0660)
Modest improvement in bias magnitude (~6% reduction), but the bias direction **flipped**: baseline favored admits under positive framing (metric=0.256), while self-help slightly favored admits under negative framing (metric=-0.190). This is a failure mode — the model over-corrected rather than eliminating bias entirely.

#### ❌ Group Attribution (-0.1070)
Self-Help made bias **worse**. Baseline metric was near-zero (-0.018, nearly unbiased), but self-help pushed it to 0.125, creating a new gender bias where the model became more likely to rate one gender differently on math ability.

#### ❌ Status Quo (-0.1766)
Self-Help substantially **worsened** status quo bias. Baseline metric (0.064) was low, indicating the model rarely defaulted to the status quo option. Self-Help pushed it to 0.240, making the model more likely to choose the status quo.

#### ❌ Anchoring (-0.0721)
Self-Help reduced decision consistency across orderings, meaning the model became **more** influenced by the order of previous decisions (more anchored), not less.

## Comparison to Paper Results

The BiasBuster paper (Echterhoff et al.) reports results for four models:
- **GPT-3.5-turbo**: Self-Help effective for framing, limited for others
- **GPT-4**: Strong improvements across all biases (selfhelp is most effective method tested)
- **Llama 2 7B**: Mixed results, some biases worsen
- **Llama 2 13B**: Similar to 7B, limited benefit

### Llama 3.1 8B vs. Paper Results
Llama 3.1 8B performance is consistent with findings for smaller open-source models:
- Only 1 of 5 biases improved (primacy)
- 3 of 5 biases worsened substantially
- 1 of 5 showed improvement but with direction flip (framing)

**Conclusion:** Self-Help is ineffective for smaller models. The technique requires larger, more capable models (GPT-4 class) to reliably debias without introducing new biases.

## Dataset Sizes
- Framing: 1000 unique prompts × 2 framings (admit/reject) = 2000 samples
- Group Attribution: 1000 unique prompts × 2 genders (male/female) = 2000 samples
- Status Quo: 1008 prompts
- Primacy: 1008 prompts (uses neutral status quo prompts)
- Anchoring: 5449 sequential decisions across 16 student sets

## Methodology

For each bias and condition (baseline vs. selfhelp):

1. **Baseline**: Send biased prompt directly to model → record answer
2. **Self-Help**: 
   - Send biased prompt to debiaser → get rewritten prompt
   - Send rewritten prompt to model → record answer
3. **Metrics**: Compute bias metric for each condition and compute delta

For anchoring (sequential bias):
- Baseline: Model evaluates students sequentially, building up conversation history
- Self-Help: After baseline session, debiaser reviews all decisions and revises any that appear anchoring-influenced

## Configuration

```yaml
debiaser:
  llm_provider: "groq"
  llm_model: "llama-3.1-8b-instant"

model:
  llm_provider: "groq"
  llm_model: "llama-3.1-8b-instant"
```

Both components use the same model (as per Self-Help methodology from the paper).

## Recommendations

For further experiments:
1. **Run with larger models** (GPT-4, Claude Opus) to replicate paper results
2. **Compare with other mitigation techniques** (awareness prompting, contrastive examples) to see if they work better for small models
3. **Analyze failure modes** — extract examples where self-help worsened bias to understand what went wrong
4. **Consider model-specific prompts** — current debiaser prompt may not be optimal for llama; fine-tuning prompt wording might help

## References

Echterhoff, J., Liu, Y., Alessa, A., McAuley, J., & He, Z. (2024). Cognitive Bias in Decision-Making with LLMs. *Findings of the Association for Computational Linguistics: EMNLP 2024*. https://huggingface.co/datasets/jecht/cognitive_bias
