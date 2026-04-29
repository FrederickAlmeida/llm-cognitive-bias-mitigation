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


_CSV_FIELDS = [
    "bias_type", "condition", "prompt_id", "sub_condition",
    "original_prompt", "debiased_prompt", "raw_answer",
    "parsed_answer", "cost_usd",
]


def save_results(results: list[PromptResult], path: str) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(_result_to_row(r))


def _result_to_row(r: PromptResult) -> dict:
    return {
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


def _append_results(path: str, results: list[PromptResult]) -> None:
    """Append results to a CSV, creating it with a header if it doesn't exist."""
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(path).exists() and Path(path).stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(_result_to_row(r))


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


def _load_processed_keys(path: str) -> set[tuple[int, str]]:
    """Load (prompt_id, sub_condition) pairs already saved in a CSV file."""
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(int(r["prompt_id"]), r["sub_condition"]) for r in csv.DictReader(f)}


def _load_processed_set_ids(path: str) -> set[int]:
    """Load unique prompt_ids (used as set_ids for anchoring)."""
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {int(r["prompt_id"]) for r in csv.DictReader(f)}


def _filter_bias_data(data, bias: str, keep_ids: set[int]):
    """Return a copy of the relevant bias list filtered to only keep_ids."""
    if bias == "anchoring":
        return [s for s in data.anchoring if s.set_id in keep_ids]
    lists = {
        "framing": data.framing,
        "group_attribution": data.group_attribution,
        "status_quo": data.status_quo,
        "primacy": data.primacy,
    }
    return [p for p in lists[bias] if p.id in keep_ids]


def _exclude_bias_data(data, bias: str, exclude_keys):
    """Return bias data excluding already-processed items.

    For anchoring: exclude_keys is a set of set_ids (int).
    For others: exclude_keys is a set of (prompt_id, sub_condition) tuples.
    """
    if bias == "anchoring":
        return [s for s in data.anchoring if s.set_id not in exclude_keys]
    lists = {
        "framing": data.framing,
        "group_attribution": data.group_attribution,
        "status_quo": data.status_quo,
        "primacy": data.primacy,
    }
    # For framing/group_attribution: sub_condition = p.framing
    # For status_quo/primacy: sub_condition = ""
    sub_fn = (lambda p: p.framing) if bias in ("framing", "group_attribution") else (lambda p: "")
    return [p for p in lists[bias] if (p.id, sub_fn(p)) not in exclude_keys]


def _get_bias_total(data, bias: str) -> int:
    """Return total number of items for a bias type."""
    counts = {
        "framing": len(data.framing),
        "group_attribution": len(data.group_attribution),
        "status_quo": len(data.status_quo),
        "primacy": len(data.primacy),
        "anchoring": len(data.anchoring),
    }
    return counts.get(bias, 0)


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

    run_dispatch = {
        "framing": runner.run_framing,
        "group_attribution": runner.run_group_attribution,
        "status_quo": runner.run_status_quo,
        "primacy": runner.run_primacy,
        "anchoring": runner.run_anchoring,
    }

    data_dispatch = {
        "framing": data.framing,
        "group_attribution": data.group_attribution,
        "status_quo": data.status_quo,
        "primacy": data.primacy,
        "anchoring": data.anchoring,
    }

    for bias in biases:
        if bias not in run_dispatch:
            print(f"[WARN] Unknown bias type '{bias}', skipping.")
            continue

        prefix = os.path.join(results_dir, f"{model_slug}_{bias}")
        baseline_path = f"{prefix}_baseline.csv"
        selfhelp_path = f"{prefix}_selfhelp.csv"

        if args.retry_failed:
            # Retry-failed: re-run only prompts with empty raw_answer
            failed_ids = _load_failed_ids(prefix)
            if not failed_ids:
                print(f"\nSkipping {bias} — no failed cases found.")
                continue
            print(f"\nRetrying {bias} ({len(failed_ids)} failed prompt IDs)...")
            bias_data = _filter_bias_data(data, bias, failed_ids)
            baseline, selfhelp = run_dispatch[bias](bias_data)
            _merge_results(baseline_path, baseline, failed_ids)
            _merge_results(selfhelp_path, selfhelp, failed_ids)
            full_baseline = _load_results_from_csv(baseline_path)
            full_selfhelp = _load_results_from_csv(selfhelp_path)
            metrics = runner.compute_metrics(bias, full_baseline, full_selfhelp)
        else:
            # Normal run with checkpoint/resume
            total = _get_bias_total(data, bias)
            if bias == "anchoring":
                processed_keys = _load_processed_set_ids(baseline_path)
                processed_count = len(processed_keys)
            else:
                processed_keys = _load_processed_keys(baseline_path)
                processed_count = len(processed_keys)

            if processed_keys and processed_count >= total:
                print(f"\n{bias} already complete ({processed_count} items). Skipping.")
                full_baseline = _load_results_from_csv(baseline_path)
                full_selfhelp = _load_results_from_csv(selfhelp_path)
                metrics = runner.compute_metrics(bias, full_baseline, full_selfhelp)
            else:
                if processed_keys:
                    remaining = total - processed_count
                    print(f"\nResuming {bias} — {processed_count}/{total} done, "
                          f"{remaining} remaining...")
                    bias_data = _exclude_bias_data(data, bias, processed_keys)
                else:
                    print(f"\nRunning {bias}...")
                    # Fresh run — remove any stale partial files
                    for p in [baseline_path, selfhelp_path]:
                        if Path(p).exists():
                            Path(p).unlink()
                    bias_data = data_dispatch[bias]

                def _make_checkpoint(bp, sp):
                    def _fn(new_b, new_sh):
                        _append_results(bp, new_b)
                        _append_results(sp, new_sh)
                    return _fn

                baseline, selfhelp = run_dispatch[bias](
                    bias_data,
                    on_checkpoint=_make_checkpoint(baseline_path, selfhelp_path),
                    checkpoint_every=20,
                )

                # Load full results (checkpoint + any previous) for metrics
                full_baseline = _load_results_from_csv(baseline_path)
                full_selfhelp = _load_results_from_csv(selfhelp_path)
                metrics = runner.compute_metrics(bias, full_baseline, full_selfhelp)

        all_baseline.extend(full_baseline)
        all_selfhelp.extend(full_selfhelp)
        all_metrics.append(metrics)

        print(f"  baseline={metrics.baseline_metric:.4f}  "
              f"selfhelp={metrics.selfhelp_metric:.4f}  "
              f"n={metrics.n_baseline}")

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
