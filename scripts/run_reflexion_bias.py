"""CLI: run the Reflexion cognitive bias debiasing pipeline.

Usage:
    uv run python scripts/run_reflexion_bias.py --model gpt-4.1-nano
    uv run python scripts/run_reflexion_bias.py --step 1 --bias framing --model gpt-4.1-nano
    uv run python scripts/run_reflexion_bias.py --step 2 --bias framing --model gpt-4.1-nano
    uv run python scripts/run_reflexion_bias.py --step 1 --bias framing --model gpt-4.1-nano --limit 20

Input per step:
  - Step 1: loads {results_dir}/{model_slug}_{bias}_baseline.csv
  - Step N≥2: loads {results_dir}/{model_slug}_{bias}_reflexion_{N-1}.csv

Output: {results_dir}/{model_slug}_{bias}_reflexion_{step}.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reflexion.llm import make_client
from src.reflexion_bias.runner import ReflexionBiasRunner, ReflexionStepResult
from src.selfhelp.runner import BiasMetrics

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reflexion cognitive bias debiasing")
    p.add_argument("--config", default="config/reflexion_bias.yaml")
    p.add_argument("--step", type=int, default=1, help="Which Reflexion step to run (1, 2, ...)")
    p.add_argument("--bias", default=None, help="Run only this bias type")
    p.add_argument("--model", default=None, help="Override actor model from config")
    p.add_argument("--limit", type=int, default=None, help="Cap unique pairs/sets processed")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_runner(cfg: dict, actor_model: str, prompts_dir: str) -> ReflexionBiasRunner:
    actor_llm = make_client(
        provider=cfg["actor"]["llm_provider"],
        model=actor_model,
    )
    reflection_llm = make_client(
        provider=cfg["reflection"]["llm_provider"],
        model=cfg["reflection"]["llm_model"],
    )
    prompts_path = os.path.join(prompts_dir, "reflexion_bias_prompts.yaml")
    return ReflexionBiasRunner(actor_llm, reflection_llm, prompts_path)


def _model_slug(model_name: str) -> str:
    slug = model_name.rsplit("/", 1)[-1]
    for ch in r'\/:*?"<>| ':
        slug = slug.replace(ch, "-")
    return slug


# ── CSV I/O ───────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "bias_type", "step", "prompt_id", "sub_condition", "original_prompt",
    "prior_raw_answer", "reflection_text", "raw_answer", "parsed_answer", "cost_usd",
]


def _load_csv(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_csv(path: str, results: list[ReflexionStepResult]) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(path).exists() and Path(path).stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "bias_type": r.bias_type,
                "step": r.step,
                "prompt_id": r.prompt_id,
                "sub_condition": r.sub_condition,
                "original_prompt": r.original_prompt,
                "prior_raw_answer": r.prior_raw_answer,
                "reflection_text": r.reflection_text,
                "raw_answer": r.raw_answer,
                "parsed_answer": r.parsed_answer,
                "cost_usd": f"{r.total_cost_usd:.8f}",
            })


def _load_reflexion_results(path: str) -> list[ReflexionStepResult]:
    """Reconstruct ReflexionStepResult objects from a saved CSV."""
    from src.reflexion.llm.base import TokenUsage
    results = []
    for row in _load_csv(path):
        cost = float(row["cost_usd"]) if row.get("cost_usd") else 0.0
        results.append(ReflexionStepResult(
            bias_type=row["bias_type"],
            step=int(row["step"]),
            prompt_id=int(row["prompt_id"]),
            sub_condition=row["sub_condition"],
            original_prompt=row["original_prompt"],
            prior_raw_answer=row["prior_raw_answer"],
            reflection_text=row.get("reflection_text", ""),
            raw_answer=row["raw_answer"],
            parsed_answer=row["parsed_answer"],
            usage=[TokenUsage(0, 0, 0, "", cost)],
        ))
    return results


# ── Checkpoint/resume helpers ─────────────────────────────────────────────────

def _processed_prompt_ids(path: str) -> set[int]:
    """Return unique prompt_ids already saved (for framing/group_attribution)."""
    return {int(r["prompt_id"]) for r in _load_csv(path)}


def _processed_set_ids(path: str) -> set[int]:
    """Return unique set_ids already saved (for anchoring)."""
    result = set()
    for row in _load_csv(path):
        sub = row.get("sub_condition", "")
        if ":" in sub:
            result.add(int(sub.split(":")[0]))
    return result


def _exclude_by_prompt_id(rows: list[dict], done: set[int]) -> list[dict]:
    return [r for r in rows if int(r["prompt_id"]) not in done]


def _exclude_by_set_id(rows: list[dict], done: set[int]) -> list[dict]:
    return [r for r in rows if int(r["sub_condition"].split(":")[0]) not in done]


# ── Prior-reflection loaders (for cumulative memory in step N≥2) ──────────────

def _prior_reflections_framing_ga(path: str) -> dict[int, str]:
    """Return {prompt_id: first non-empty reflection_text} from a step N-1 CSV."""
    result: dict[int, str] = {}
    for row in _load_csv(path):
        pid = int(row["prompt_id"])
        refl = row.get("reflection_text", "").strip()
        if pid not in result and refl:
            result[pid] = refl
    return result


def _prior_reflections_anchoring(path: str) -> dict[int, str]:
    """Return {set_id: first non-empty reflection_text} from a step N-1 CSV."""
    result: dict[int, str] = {}
    for row in _load_csv(path):
        sub = row.get("sub_condition", "")
        if ":" not in sub:
            continue
        set_id = int(sub.split(":")[0])
        refl = row.get("reflection_text", "").strip()
        if set_id not in result and refl:
            result[set_id] = refl
    return result


# ── Limit helpers ─────────────────────────────────────────────────────────────

def _apply_limit(bias: str, rows: list[dict], limit: int | None) -> list[dict]:
    """Cap by unique pairs/sets, keeping ALL sub_conditions for each included ID."""
    if limit is None:
        return rows
    if bias == "anchoring":
        seen_sets: set[int] = set()
        filtered = []
        for row in rows:
            set_id = int(row["sub_condition"].split(":")[0])
            if set_id not in seen_sets:
                if len(seen_sets) >= limit:
                    continue
                seen_sets.add(set_id)
            filtered.append(row)
        return filtered
    else:
        seen_pids: set[int] = set()
        filtered = []
        for row in rows:
            pid = int(row["prompt_id"])
            if pid not in seen_pids:
                if len(seen_pids) >= limit:
                    continue
                seen_pids.add(pid)
            filtered.append(row)
        return filtered


# ── Metrics display ───────────────────────────────────────────────────────────

def print_metrics_table(all_metrics: list[BiasMetrics], step: int) -> None:
    print("\n" + "=" * 65)
    print(f"{'Bias':<20} {'Baseline':>10} {f'Reflexion-{step}':>13} {'Improvement':>12}")
    print("-" * 65)
    for m in all_metrics:
        sign = "+" if m.delta >= 0 else ""
        print(
            f"{m.bias_type:<20} "
            f"{m.baseline_metric:>10.4f} "
            f"{m.selfhelp_metric:>13.4f} "
            f"{sign}{m.delta:>11.4f}"
        )
    print("=" * 65)
    print("(Improvement: positive = less bias with Reflexion)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    step = args.step
    biases = cfg["dataset"]["biases"]
    if args.bias:
        biases = [args.bias]

    actor_model = args.model or cfg["actor"]["llm_model"]
    model_slug = _model_slug(actor_model)
    results_dir = cfg["output"]["results_dir"]
    prompts_dir = cfg.get("prompts_dir", "prompts/")

    print(f"Building runner (actor={actor_model}, "
          f"reflection={cfg['reflection']['llm_model']})...")
    runner = build_runner(cfg, actor_model, prompts_dir)

    all_metrics: list[BiasMetrics] = []
    total_cost = 0.0

    run_dispatch = {
        "framing": runner.run_framing,
        "group_attribution": runner.run_group_attribution,
        "anchoring": runner.run_anchoring,
        "primacy": runner.run_primacy,
        "status_quo": runner.run_status_quo,
    }

    for bias in biases:
        if bias not in run_dispatch:
            print(f"[WARN] Bias '{bias}' not implemented for Reflexion yet, skipping.")
            continue

        prefix = os.path.join(results_dir, f"{model_slug}_{bias}")
        baseline_path = f"{prefix}_baseline.csv"
        input_path = baseline_path if step == 1 else f"{prefix}_reflexion_{step - 1}.csv"
        output_path = f"{prefix}_reflexion_{step}.csv"

        if not Path(input_path).exists():
            print(f"\n[WARN] Input not found for {bias}: {input_path}")
            continue

        # Load input rows (with limit applied)
        all_input_rows = _load_csv(input_path)
        all_input_rows = _apply_limit(bias, all_input_rows, args.limit)

        # Load prior reflections for cumulative memory (step N≥2)
        prior_reflections: dict = {}
        if step >= 2:
            prior_path = f"{prefix}_reflexion_{step - 1}.csv"
            if bias == "anchoring":
                prior_reflections = _prior_reflections_anchoring(prior_path)
            else:
                prior_reflections = _prior_reflections_framing_ga(prior_path)

        # Check how many pairs/sets are already done (resume)
        if bias == "anchoring":
            done_keys = _processed_set_ids(output_path)
            remaining_rows = _exclude_by_set_id(all_input_rows, done_keys)
        else:
            done_keys = _processed_prompt_ids(output_path)
            remaining_rows = _exclude_by_prompt_id(all_input_rows, done_keys)

        if done_keys and not remaining_rows:
            print(f"\n{bias} reflexion_{step} already complete ({len(done_keys)} items). Skipping.")
        else:
            if done_keys:
                print(f"\nResuming {bias} step {step} — {len(done_keys)} done, "
                      f"{len(remaining_rows)} rows remaining...")
            else:
                print(f"\nRunning {bias} step {step}...")

            def make_checkpoint(path):
                def fn(new_results):
                    _append_csv(path, new_results)
                return fn

            kwargs = dict(
                step=step,
                prior_reflections=prior_reflections,
                on_checkpoint=make_checkpoint(output_path),
                checkpoint_every=1 if bias == "anchoring" else 20,
            )
            run_dispatch[bias](remaining_rows, **kwargs)

        # Load full output for metrics
        reflexion_results = _load_reflexion_results(output_path)
        baseline_rows = _load_csv(baseline_path)
        if not baseline_rows or not reflexion_results:
            print(f"  [WARN] Missing data for {bias} metrics, skipping.")
            continue

        metrics = runner.compute_metrics(bias, baseline_rows, reflexion_results)
        all_metrics.append(metrics)
        step_cost = sum(r.total_cost_usd for r in reflexion_results)
        total_cost += step_cost

        print(f"  baseline={metrics.baseline_metric:.4f}  "
              f"reflexion_{step}={metrics.selfhelp_metric:.4f}  "
              f"n={metrics.n_baseline}")

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{prefix}_reflexion_{step}_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "bias_type": metrics.bias_type,
                "step": step,
                "baseline_metric": metrics.baseline_metric,
                f"reflexion_{step}_metric": metrics.selfhelp_metric,
                "delta": metrics.delta,
                "n_baseline": metrics.n_baseline,
                "n_reflexion": metrics.n_selfhelp,
            }, f, indent=2)

    if all_metrics:
        print_metrics_table(all_metrics, step)
    print(f"\nTotal cost this run: ${total_cost:.4f} USD")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
