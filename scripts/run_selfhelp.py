"""CLI: run the Self-Help debiasing evaluation pipeline.

Usage:
    uv run python scripts/run_selfhelp.py
    uv run python scripts/run_selfhelp.py --config config/selfhelp.yaml --bias framing --limit 20
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

from datasets.bias_buster_loader import BiasBusterLoader
from src.reflexion.llm import make_client
from src.selfhelp.debiaser import SelfHelpDebiaser
from src.selfhelp.runner import BiasMetrics, PromptResult, SelfHelpRunner

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Help cognitive bias evaluation")
    p.add_argument("--config", default="config/selfhelp.yaml")
    p.add_argument("--bias", default=None, help="Run only this bias type")
    p.add_argument("--limit", type=int, default=None, help="Cap samples per bias")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only prompts whose raw_answer was empty in existing result files",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_runner(cfg: dict, prompts_dir: str) -> SelfHelpRunner:
    debiaser_llm = make_client(
        provider=cfg["debiaser"]["llm_provider"],
        model=cfg["debiaser"]["llm_model"],
    )
    model_llm = make_client(
        provider=cfg["model"]["llm_provider"],
        model=cfg["model"]["llm_model"],
    )
    prompts_path = os.path.join(prompts_dir, "selfhelp_prompts.yaml")
    debiaser = SelfHelpDebiaser(debiaser_llm, prompts_path)
    return SelfHelpRunner(model=model_llm, debiaser=debiaser)


def save_results(results: list[PromptResult], path: str) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "bias_type", "condition", "prompt_id", "sub_condition",
            "original_prompt", "debiased_prompt", "raw_answer",
            "parsed_answer", "cost_usd",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "bias_type": r.bias_type,
                "condition": r.condition,
                "prompt_id": r.prompt_id,
                "sub_condition": r.sub_condition,
                "original_prompt": r.original_prompt,
                "debiased_prompt": r.debiased_prompt,
                "raw_answer": r.raw_answer,
                "parsed_answer": r.parsed_answer,
                "cost_usd": f"{r.total_cost_usd:.8f}",
            })


def print_metrics_table(all_metrics: list[BiasMetrics]) -> None:
    print("\n" + "=" * 60)
    print(f"{'Bias':<20} {'Baseline':>10} {'Self-Help':>10} {'Improvement':>12}")
    print("-" * 60)
    for m in all_metrics:
        improvement = m.delta
        sign = "+" if improvement >= 0 else ""
        print(
            f"{m.bias_type:<20} "
            f"{m.baseline_metric:>10.4f} "
            f"{m.selfhelp_metric:>10.4f} "
            f"{sign}{improvement:>11.4f}"
        )
    print("=" * 60)
    print("(Improvement: positive = less bias with self-help)")


def _load_failed_ids(prefix: str) -> set[int]:
    """Return prompt_ids from <prefix>_baseline.csv where raw_answer is empty."""
    path = f"{prefix}_baseline.csv"
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(r["prompt_id"]) for r in reader if not r["raw_answer"].strip()}


def _filter_bias_data(data, bias: str, failed_ids: set[int]):
    """Return a copy of the relevant bias list filtered to only failed_ids."""
    if bias == "anchoring":
        return [s for s in data.anchoring if s.set_id in failed_ids]
    lists = {
        "framing": data.framing,
        "group_attribution": data.group_attribution,
        "status_quo": data.status_quo,
        "primacy": data.primacy,
    }
    return [p for p in lists[bias] if p.id in failed_ids]


def _merge_results(path: str, new_results: list[PromptResult], failed_ids: set[int]) -> None:
    """Replace rows with failed_ids in an existing CSV with new_results, then save."""
    existing: list[dict] = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if int(r["prompt_id"]) not in failed_ids]

    fieldnames = [
        "bias_type", "condition", "prompt_id", "sub_condition",
        "original_prompt", "debiased_prompt", "raw_answer", "parsed_answer", "cost_usd",
    ]
    new_rows = [
        {
            "bias_type": r.bias_type,
            "condition": r.condition,
            "prompt_id": r.prompt_id,
            "sub_condition": r.sub_condition,
            "original_prompt": r.original_prompt,
            "debiased_prompt": r.debiased_prompt,
            "raw_answer": r.raw_answer,
            "parsed_answer": r.parsed_answer,
            "cost_usd": f"{r.total_cost_usd:.8f}",
        }
        for r in new_results
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(new_rows)


def _load_results_from_csv(path: str) -> list[PromptResult]:
    """Reload PromptResult objects from a saved CSV for metric recomputation."""
    from src.reflexion.llm.base import TokenUsage
    results = []
    if not Path(path).exists():
        return results
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cost = float(r["cost_usd"]) if r["cost_usd"] else 0.0
            results.append(PromptResult(
                bias_type=r["bias_type"],
                condition=r["condition"],
                prompt_id=int(r["prompt_id"]),
                sub_condition=r["sub_condition"],
                original_prompt=r["original_prompt"],
                debiased_prompt=r["debiased_prompt"],
                raw_answer=r["raw_answer"],
                parsed_answer=r["parsed_answer"],
                usage=[TokenUsage(0, 0, 0, "", cost)],
            ))
    return results


def _model_slug(model_name: str) -> str:
    """Sanitize a model name for use in filenames (e.g. 'openai/gpt-oss-20b' → 'gpt-oss-20b')."""
    # Drop provider prefix (anything before the last '/')
    slug = model_name.rsplit("/", 1)[-1]
    # Replace any remaining unsafe characters
    for ch in r'\/:*?"<>| ':
        slug = slug.replace(ch, "-")
    return slug


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    limit = args.limit or cfg["dataset"].get("limit")
    biases = cfg["dataset"]["biases"]
    if args.bias:
        biases = [args.bias]

    results_dir = cfg["output"]["results_dir"]
    prompts_dir = cfg.get("prompts_dir", "prompts/")
    model_slug = _model_slug(cfg["model"]["llm_model"])

    print(f"Loading dataset (limit={limit})...")
    loader = BiasBusterLoader()
    data = loader.load(limit=limit)

    print(f"Building runner (debiaser={cfg['debiaser']['llm_model']}, "
          f"model={cfg['model']['llm_model']})...")
    runner = build_runner(cfg, prompts_dir)

    all_baseline: list[PromptResult] = []
    all_selfhelp: list[PromptResult] = []
    all_metrics: list[BiasMetrics] = []

    bias_dispatch = {
        "framing": lambda: runner.run_framing(data.framing),
        "group_attribution": lambda: runner.run_group_attribution(data.group_attribution),
        "status_quo": lambda: runner.run_status_quo(data.status_quo),
        "primacy": lambda: runner.run_primacy(data.primacy),
        "anchoring": lambda: runner.run_anchoring(data.anchoring),
    }

    for bias in biases:
        if bias not in bias_dispatch:
            print(f"[WARN] Unknown bias type '{bias}', skipping.")
            continue

        prefix = os.path.join(results_dir, f"{model_slug}_{bias}")

        if args.retry_failed:
            failed_ids = _load_failed_ids(prefix)
            if not failed_ids:
                print(f"\nSkipping {bias} — no failed cases found.")
                continue
            print(f"\nRetrying {bias} ({len(failed_ids)} failed prompt IDs)...")
            filtered = _filter_bias_data(data, bias, failed_ids)
            bias_dispatch[bias] = {
                "framing": lambda f=filtered: runner.run_framing(f),
                "group_attribution": lambda f=filtered: runner.run_group_attribution(f),
                "status_quo": lambda f=filtered: runner.run_status_quo(f),
                "primacy": lambda f=filtered: runner.run_primacy(f),
                "anchoring": lambda f=filtered: runner.run_anchoring(f),
            }[bias]
        else:
            print(f"\nRunning {bias}...")

        baseline, selfhelp = bias_dispatch[bias]()
        metrics = runner.compute_metrics(bias, baseline, selfhelp)

        all_baseline.extend(baseline)
        all_selfhelp.extend(selfhelp)
        all_metrics.append(metrics)

        print(f"  baseline={metrics.baseline_metric:.4f}  "
              f"selfhelp={metrics.selfhelp_metric:.4f}  "
              f"n={metrics.n_baseline}")

        if args.retry_failed:
            _merge_results(f"{prefix}_baseline.csv", baseline, failed_ids)
            _merge_results(f"{prefix}_selfhelp.csv", selfhelp, failed_ids)
            # Recompute metrics from the full merged files
            full_baseline = _load_results_from_csv(f"{prefix}_baseline.csv")
            full_selfhelp = _load_results_from_csv(f"{prefix}_selfhelp.csv")
            metrics = runner.compute_metrics(bias, full_baseline, full_selfhelp)
        else:
            save_results(baseline, f"{prefix}_baseline.csv")
            save_results(selfhelp, f"{prefix}_selfhelp.csv")

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "bias_type": metrics.bias_type,
                    "baseline_metric": metrics.baseline_metric,
                    "selfhelp_metric": metrics.selfhelp_metric,
                    "delta": metrics.delta,
                    "n_baseline": metrics.n_baseline,
                    "n_selfhelp": metrics.n_selfhelp,
                },
                f,
                indent=2,
            )

    total_cost = sum(r.total_cost_usd for r in all_baseline + all_selfhelp)
    print_metrics_table(all_metrics)
    print(f"\nTotal cost: ${total_cost:.4f} USD")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
