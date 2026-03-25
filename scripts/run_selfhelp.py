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
from datetime import datetime
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
                "original_prompt": r.original_prompt[:500],  # truncate for readability
                "debiased_prompt": r.debiased_prompt[:500],
                "raw_answer": r.raw_answer[:300],
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    limit = args.limit or cfg["dataset"].get("limit")
    biases = cfg["dataset"]["biases"]
    if args.bias:
        biases = [args.bias]

    results_dir = cfg["output"]["results_dir"]
    prompts_dir = cfg.get("prompts_dir", "prompts/")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        print(f"\nRunning {bias}...")
        baseline, selfhelp = bias_dispatch[bias]()
        metrics = runner.compute_metrics(bias, baseline, selfhelp)

        all_baseline.extend(baseline)
        all_selfhelp.extend(selfhelp)
        all_metrics.append(metrics)

        print(f"  baseline={metrics.baseline_metric:.4f}  "
              f"selfhelp={metrics.selfhelp_metric:.4f}  "
              f"n={metrics.n_baseline}")

    # Save outputs
    baseline_path = os.path.join(results_dir, f"run_{timestamp}_baseline.csv")
    selfhelp_path = os.path.join(results_dir, f"run_{timestamp}_selfhelp.csv")
    metrics_path = os.path.join(results_dir, f"run_{timestamp}_metrics.json")

    save_results(all_baseline, baseline_path)
    save_results(all_selfhelp, selfhelp_path)

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "bias_type": m.bias_type,
                    "baseline_metric": m.baseline_metric,
                    "selfhelp_metric": m.selfhelp_metric,
                    "delta": m.delta,
                    "n_baseline": m.n_baseline,
                    "n_selfhelp": m.n_selfhelp,
                }
                for m in all_metrics
            ],
            f,
            indent=2,
        )

    total_cost = sum(r.total_cost_usd for r in all_baseline + all_selfhelp)
    print_metrics_table(all_metrics)
    print(f"\nTotal cost: ${total_cost:.4f} USD")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
