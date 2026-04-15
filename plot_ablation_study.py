"""
Plot the SAHL ablation study results (weight vs metrics).

Reads ablation_summary.csv and produces a publication-ready PDF plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_ablation_study(csv_path: Path, output_path: Path) -> None:
    """
    Load ablation CSV and plot F1, Critical Recall, and Precision vs weight.
    
    Args:
        csv_path: Path to ablation_summary.csv
        output_path: Path to save the PDF (e.g., ablation_study.pdf)
    """
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        df["weight"],
        df["f1"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="F1-Score @ 0.65",
    )
    ax.plot(
        df["weight"],
        df["critical_recall"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="Critical Recall",
    )
    ax.plot(
        df["weight"],
        df["precision"],
        marker="^",
        linewidth=2,
        markersize=8,
        label="Precision @ 0.65",
    )
    
    ax.set_xlabel("Asymmetric Weight (multiplier)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("SAHL Weight Ablation Study", fontsize=14, fontweight="bold")
    ax.set_xticks(df["weight"])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.set_ylim(0, 1.0)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nAblation plot saved to: {output_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready plot of SAHL weight ablation."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="results/ablation_summary.csv",
        help="Path to ablation_summary.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ablation_study.pdf",
        help="Path to save the output PDF.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    csv_path = Path(args.csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    output_path = Path(args.output)
    plot_ablation_study(csv_path, output_path)


if __name__ == "__main__":
    main()
