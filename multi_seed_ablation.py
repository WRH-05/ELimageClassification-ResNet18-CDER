"""Run and summarize the true multi-seed SAHL weight ablation study.

This script orchestrates 4 SAHL weights across 3 fixed seeds, reuses the
existing training/export/report pipeline, and aggregates the resulting test
reports into a publication-ready Markdown table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from aggregate_results import (
    compute_critical_recall,
    compute_mae,
    compute_metrics_at_threshold,
    load_test_report,
)
from experiment_runner import ExperimentConfig, run_single_experiment


WEIGHTS: Sequence[float] = (1.0, 1.5, 2.5, 5.0)
SEEDS: Sequence[int] = (42, 123, 2026)
GENERAL_THRESHOLD = 0.65
CRITICAL_PREDICTION_THRESHOLD = 0.6
CRITICAL_TARGET_THRESHOLD = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the true multi-seed SAHL weight ablation study."
    )
    parser.add_argument("--csv_path", type=str, default="labels.csv")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--precision_floor",
        type=float,
        default=0.0,
        help="Checkpoint precision floor for this ablation run; defaults to 0.0 so all seeds complete.",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="multi_seed_ablation_summary.md",
        help="Markdown summary output path.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace, seed: int, weight: float) -> ExperimentConfig:
    return ExperimentConfig(
        seed=seed,
        loss_type="weighted_l1",
        loss_weight_multiplier=weight,
        loss_weight_threshold=0.66,
        critical_recall_threshold=CRITICAL_PREDICTION_THRESHOLD,
        critical_target_threshold=CRITICAL_TARGET_THRESHOLD,
        precision_floor=args.precision_floor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        csv_path=args.csv_path,
        data_root=args.data_root,
        image_size=args.image_size,
        device=args.device,
    )


def report_path_for(seed: int, weight: float) -> Path:
    return Path("testCsv") / f"test_split_report_seed{seed}_weighted_l1_w{weight:.1f}.csv"


def format_mean_std(values: Sequence[float], decimals: int) -> str:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def summarize_reports() -> Tuple[List[Dict[str, str]], Dict[float, Dict[str, List[float]]]]:
    summary_rows: List[Dict[str, str]] = []
    raw_metrics: Dict[float, Dict[str, List[float]]] = {
        weight: {"f1": [], "precision": [], "recall": [], "critical_recall": [], "mae": []}
        for weight in WEIGHTS
    }

    for weight in WEIGHTS:
        for seed in SEEDS:
            report_path = report_path_for(seed, weight)
            if not report_path.exists():
                raise FileNotFoundError(
                    f"Expected test report not found: {report_path}. "
                    "Run the training/export pipeline first."
                )

            predictions, targets = load_test_report(report_path)
            threshold_metrics = compute_metrics_at_threshold(
                predictions,
                targets,
                prediction_threshold=GENERAL_THRESHOLD,
                target_threshold=GENERAL_THRESHOLD,
            )

            raw_metrics[weight]["f1"].append(threshold_metrics["f1"])
            raw_metrics[weight]["precision"].append(threshold_metrics["precision"])
            raw_metrics[weight]["recall"].append(threshold_metrics["recall"])
            raw_metrics[weight]["critical_recall"].append(
                compute_critical_recall(
                    predictions,
                    targets,
                    prediction_threshold=CRITICAL_PREDICTION_THRESHOLD,
                    target_threshold=CRITICAL_TARGET_THRESHOLD,
                )
            )
            raw_metrics[weight]["mae"].append(compute_mae(predictions, targets))

        summary_rows.append(
            {
                "weight": f"{weight:.1f}x",
                "f1": format_mean_std(raw_metrics[weight]["f1"], 3),
                "precision": format_mean_std(raw_metrics[weight]["precision"], 3),
                "recall": format_mean_std(raw_metrics[weight]["recall"], 3),
                "critical_recall": format_mean_std(raw_metrics[weight]["critical_recall"], 3),
                "mae": format_mean_std(raw_metrics[weight]["mae"], 4),
            }
        )

    return summary_rows, raw_metrics


def paired_p_value(sample_a: Sequence[float], sample_b: Sequence[float]) -> Tuple[str, float]:
    ttest = stats.ttest_rel(sample_a, sample_b)
    p_value = float(ttest.pvalue)
    method = "ttest_rel"

    if math.isnan(p_value):
        wilcoxon = stats.wilcoxon(sample_a, sample_b, zero_method="wilcox", alternative="two-sided")
        p_value = float(wilcoxon.pvalue)
        method = "wilcoxon"

    return method, p_value


def write_markdown_summary(output_path: Path, summary_rows: Sequence[Dict[str, str]]) -> None:
    lines = [
        "# Multi-Seed SAHL Weight Ablation Summary",
        "",
        "Metrics are reported as Mean ± Std across seeds [42, 123, 2026].",
        f"General metrics use threshold {GENERAL_THRESHOLD:.2f}; critical recall uses pred >= {CRITICAL_PREDICTION_THRESHOLD:.1f} and target >= {CRITICAL_TARGET_THRESHOLD:.1f}.",
        "",
        "| Weight | F1-Score | Precision | Recall | Critical Recall | MAE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['weight']} | {row['f1']} | {row['precision']} | {row['recall']} | {row['critical_recall']} | {row['mae']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("Running true multi-seed SAHL weight ablation...")
    for weight in WEIGHTS:
        for seed in SEEDS:
            print(f"  Training weight={weight:.1f} seed={seed}")
            run_single_experiment(make_config(args, seed=seed, weight=weight))

    summary_rows, raw_metrics = summarize_reports()

    method_critical, p_critical = paired_p_value(
        raw_metrics[1.0]["critical_recall"],
        raw_metrics[1.5]["critical_recall"],
    )
    method_mae, p_mae = paired_p_value(raw_metrics[1.0]["mae"], raw_metrics[1.5]["mae"])

    print("\nPaired significance tests: 1.0x vs 1.5x")
    print(
        f"  Critical Recall: {method_critical} p-value = {p_critical:.6f} "
        f"({'significant' if p_critical < 0.05 else 'not significant'})"
    )
    print(
        f"  MAE: {method_mae} p-value = {p_mae:.6f} "
        f"({'significant' if p_mae < 0.05 else 'not significant'})"
    )

    output_path = Path(args.output_md)
    write_markdown_summary(output_path, summary_rows)
    print(f"\nMarkdown summary written to: {output_path}")
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()