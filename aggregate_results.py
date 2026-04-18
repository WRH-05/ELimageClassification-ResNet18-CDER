"""
Aggregate multi-seed variance and ablation results into summary tables.

Reads test report CSVs and computes Mean ± StdDev metrics for publication.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def load_test_report(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a test report CSV and return predictions and targets."""
    df = pd.read_csv(csv_path)
    predictions = df["prediction"].to_numpy(dtype=np.float64)
    targets = df["target"].to_numpy(dtype=np.float64)
    return predictions, targets


def compute_metrics_at_threshold(
    predictions: np.ndarray,
    targets: np.ndarray,
    prediction_threshold: float = 0.65,
    target_threshold: float = 0.65,
) -> Dict[str, float]:
    """Compute F1, precision, recall at a given threshold."""
    y_true_binary = (targets >= target_threshold).astype(int)
    y_pred_binary = (predictions >= prediction_threshold).astype(int)

    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def compute_critical_recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    prediction_threshold: float = 0.6,
    target_threshold: float = 0.8,
) -> float:
    """Compute critical recall (recall on high-severity cases)."""
    critical_mask = targets >= target_threshold
    critical_count = int(critical_mask.sum())
    if critical_count == 0:
        return 0.0

    true_positives = int((predictions[critical_mask] >= prediction_threshold).sum())
    return float(true_positives / critical_count)


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(predictions - targets)))


def aggregate_variance_study(v3_weight: float, v3_label: str) -> None:
    """Aggregate multi-seed variance results (V1 and configurable V3 SAHL weight)."""
    seeds = [42, 123, 2026]
    report_dir = Path("testCsv")

    results: Dict[str, List[float]] = {
        "v1_f1": [],
        "v1_precision": [],
        "v1_recall": [],
        "v1_mae": [],
        "v1_critical_recall": [],
        "v3_f1": [],
        "v3_precision": [],
        "v3_recall": [],
        "v3_mae": [],
        "v3_critical_recall": [],
    }

    for seed in seeds:
        # V1: MSE
        v1_report = report_dir / f"test_split_report_seed{seed}_mse_w1.0.csv"
        if v1_report.exists():
            preds, targets = load_test_report(v1_report)
            v1_metrics = compute_metrics_at_threshold(preds, targets, 0.65, 0.65)
            results["v1_f1"].append(v1_metrics["f1"])
            results["v1_precision"].append(v1_metrics["precision"])
            results["v1_recall"].append(v1_metrics["recall"])
            results["v1_mae"].append(compute_mae(preds, targets))
            results["v1_critical_recall"].append(compute_critical_recall(preds, targets))

        # V3.1: weighted_l1 @ configurable weight
        v3_report = report_dir / f"test_split_report_seed{seed}_weighted_l1_w{v3_weight:.1f}.csv"
        if v3_report.exists():
            preds, targets = load_test_report(v3_report)
            v3_metrics = compute_metrics_at_threshold(preds, targets, 0.65, 0.65)
            results["v3_f1"].append(v3_metrics["f1"])
            results["v3_precision"].append(v3_metrics["precision"])
            results["v3_recall"].append(v3_metrics["recall"])
            results["v3_mae"].append(compute_mae(preds, targets))
            results["v3_critical_recall"].append(compute_critical_recall(preds, targets))

    # Compute statistics
    output_lines = [
        "# Variance Study Results",
        "",
        "Multi-seed study across 3 seeds (42, 123, 2026).",
        "Metrics shown as Mean ± Standard Deviation.",
        "",
        "## F1-Score @ threshold 0.65",
        "",
    ]

    # F1 table
    if results["v1_f1"]:
        v1_f1_mean = np.mean(results["v1_f1"])
        v1_f1_std = np.std(results["v1_f1"], ddof=1) if len(results["v1_f1"]) > 1 else 0
        v3_f1_mean = np.mean(results["v3_f1"])
        v3_f1_std = np.std(results["v3_f1"], ddof=1) if len(results["v3_f1"]) > 1 else 0

        output_lines.extend([
            "| Model | F1-Score |",
            "|-------|----------|",
            f"| V1 (MSE) | ${v1_f1_mean:.3f} \\pm {v1_f1_std:.3f}$ |",
            f"| {v3_label} | ${v3_f1_mean:.3f} \\pm {v3_f1_std:.3f}$ |",
            "",
        ])

    # Precision table
    output_lines.append("## Precision @ threshold 0.65")
    output_lines.append("")
    if results["v1_precision"]:
        v1_prec_mean = np.mean(results["v1_precision"])
        v1_prec_std = np.std(results["v1_precision"], ddof=1) if len(results["v1_precision"]) > 1 else 0
        v3_prec_mean = np.mean(results["v3_precision"])
        v3_prec_std = np.std(results["v3_precision"], ddof=1) if len(results["v3_precision"]) > 1 else 0

        output_lines.extend([
            "| Model | Precision |",
            "|-------|-----------|",
            f"| V1 (MSE) | ${v1_prec_mean:.3f} \\pm {v1_prec_std:.3f}$ |",
            f"| {v3_label} | ${v3_prec_mean:.3f} \\pm {v3_prec_std:.3f}$ |",
            "",
        ])

    # Recall table
    output_lines.append("## Recall @ threshold 0.65")
    output_lines.append("")
    if results["v1_recall"]:
        v1_recall_mean = np.mean(results["v1_recall"])
        v1_recall_std = np.std(results["v1_recall"], ddof=1) if len(results["v1_recall"]) > 1 else 0
        v3_recall_mean = np.mean(results["v3_recall"])
        v3_recall_std = np.std(results["v3_recall"], ddof=1) if len(results["v3_recall"]) > 1 else 0

        output_lines.extend([
            "| Model | Recall |",
            "|-------|--------|",
            f"| V1 (MSE) | ${v1_recall_mean:.3f} \\pm {v1_recall_std:.3f}$ |",
            f"| {v3_label} | ${v3_recall_mean:.3f} \\pm {v3_recall_std:.3f}$ |",
            "",
        ])

    # MAE table
    output_lines.append("## Mean Absolute Error (MAE)")
    output_lines.append("")
    if results["v1_mae"]:
        v1_mae_mean = np.mean(results["v1_mae"])
        v1_mae_std = np.std(results["v1_mae"], ddof=1) if len(results["v1_mae"]) > 1 else 0
        v3_mae_mean = np.mean(results["v3_mae"])
        v3_mae_std = np.std(results["v3_mae"], ddof=1) if len(results["v3_mae"]) > 1 else 0

        output_lines.extend([
            "| Model | MAE |",
            "|-------|-----|",
            f"| V1 (MSE) | ${v1_mae_mean:.4f} \\pm {v1_mae_std:.4f}$ |",
            f"| {v3_label} | ${v3_mae_mean:.4f} \\pm {v3_mae_std:.4f}$ |",
            "",
        ])

    # Critical Recall table
    output_lines.append("## Critical Recall @ (pred threshold 0.6, target threshold 0.8)")
    output_lines.append("")
    if results["v1_critical_recall"]:
        v1_crit_mean = np.mean(results["v1_critical_recall"])
        v1_crit_std = np.std(results["v1_critical_recall"], ddof=1) if len(results["v1_critical_recall"]) > 1 else 0
        v3_crit_mean = np.mean(results["v3_critical_recall"])
        v3_crit_std = np.std(results["v3_critical_recall"], ddof=1) if len(results["v3_critical_recall"]) > 1 else 0

        output_lines.extend([
            "| Model | Critical Recall |",
            "|-------|-----------------|",
            f"| V1 (MSE) | ${v1_crit_mean:.3f} \\pm {v1_crit_std:.3f}$ |",
            f"| {v3_label} | ${v3_crit_mean:.3f} \\pm {v3_crit_std:.3f}$ |",
            "",
        ])

    # Write to file
    output_path = Path("results") / "variance_metrics.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nVariance study results saved to: {output_path}")
    print("\n".join(output_lines))


def aggregate_ablation_study() -> None:
    """Aggregate ablation sweep results (weight 1.0x, 1.5x, 2.5x, 5.0x)."""
    seed = 42
    weights = [1.0, 1.5, 2.5, 5.0]
    report_dir = Path("testCsv")

    results: List[Dict[str, float]] = []

    for weight in weights:
        report_path = report_dir / f"test_split_report_seed{seed}_weighted_l1_w{weight:.1f}.csv"
        if report_path.exists():
            preds, targets = load_test_report(report_path)
            metrics = compute_metrics_at_threshold(preds, targets, 0.65, 0.65)
            critical_recall = compute_critical_recall(preds, targets)
            mae = compute_mae(preds, targets)

            results.append({
                "weight": weight,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "critical_recall": critical_recall,
                "mae": mae,
            })

    if not results:
        print("No ablation reports found.")
        return

    # Write CSV
    csv_output = Path("results") / "ablation_summary.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["weight", "f1", "precision", "recall", "critical_recall", "mae"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAblation summary CSV saved to: {csv_output}")

    # Write Markdown table
    output_lines = [
        "# SAHL Weight Ablation Study (Seed 42)",
        "",
        "| Weight | F1 @ 0.65 | Precision @ 0.65 | Recall @ 0.65 | Critical Recall | MAE |",
        "|--------|-----------|------------------|---------------|-----------------|-----|",
    ]

    for row in results:
        output_lines.append(
            f"| {row['weight']:.1f}x | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['critical_recall']:.4f} | {row['mae']:.4f} |"
        )

    md_output = Path("results") / "ablation_summary.md"
    md_output.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nAblation summary Markdown saved to: {md_output}")
    print("\n".join(output_lines))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate and summarize experiment results."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="variance",
        choices=["variance", "ablation"],
        help="Which aggregation to run: 'variance' or 'ablation'.",
    )
    parser.add_argument(
        "--v3_weight",
        type=float,
        default=2.5,
        help="SAHL weighted_l1 multiplier to use for V3 variance aggregation.",
    )
    parser.add_argument(
        "--v3_label",
        type=str,
        default="V3.1 (SAHL 2.5x)",
        help="Display label for V3 model in variance output tables.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.mode == "variance":
        aggregate_variance_study(v3_weight=args.v3_weight, v3_label=args.v3_label)
    elif args.mode == "ablation":
        aggregate_ablation_study()


if __name__ == "__main__":
    main()
