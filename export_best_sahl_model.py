"""
Select and export the best 2.5x SAHL model from the multi-seed variance study.

Evaluates all variance-study checkpoints and exports the best one.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_test_report(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a test report CSV and return predictions and targets."""
    df = pd.read_csv(csv_path)
    predictions = df["prediction"].to_numpy(dtype=np.float64)
    targets = df["target"].to_numpy(dtype=np.float64)
    return predictions, targets


def compute_f1_at_threshold(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.65,
) -> float:
    """Compute F1 at a given threshold."""
    from sklearn.metrics import f1_score
    
    y_true = (targets >= threshold).astype(int)
    y_pred = (predictions >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(predictions - targets)))


def select_best_sahl_model() -> Tuple[int, float, float]:
    """
    Select the best 2.5x SAHL model from variance study runs.
    
    Returns:
        (best_seed, best_f1, best_mae)
    """
    seeds = [42, 123, 2026]
    report_dir = Path("testCsv")
    
    results: List[Tuple[int, float, float]] = []
    
    for seed in seeds:
        report_path = report_dir / f"test_split_report_seed{seed}_weighted_l1_w2.5.csv"
        if report_path.exists():
            preds, targets = load_test_report(report_path)
            f1 = compute_f1_at_threshold(preds, targets, 0.65)
            mae = compute_mae(preds, targets)
            results.append((seed, f1, mae))
            print(f"Seed {seed}: F1={f1:.4f}, MAE={mae:.4f}")
    
    if not results:
        raise RuntimeError("No 2.5x SAHL reports found in variance study.")
    
    # Sort by F1 (descending), then MAE (ascending)
    results.sort(key=lambda x: (-x[1], x[2]))
    best_seed, best_f1, best_mae = results[0]
    
    print(f"\nBest model: Seed {best_seed} (F1={best_f1:.4f}, MAE={best_mae:.4f})")
    return best_seed, best_f1, best_mae


def export_best_sahl_model() -> None:
    """
    Find the best 2.5x SAHL checkpoint and export it to ONNX.
    """
    best_seed, best_f1, best_mae = select_best_sahl_model()
    
    checkpoint_path = Path("models/pth") / f"best_model_seed{best_seed}_weighted_l1_w2.5.pth"
    onnx_output = Path("models/onnx") / "best_sahl_2.5x_final.onnx"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nExporting checkpoint: {checkpoint_path}")
    print(f"ONNX output: {onnx_output}")
    
    # Find Python executable
    python_exe = Path(".venv") / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found at {python_exe}")
    
    # Run export_model.py
    cmd = [
        str(python_exe),
        "export_model.py",
        "--checkpoint", str(checkpoint_path),
        "--onnx_output", str(onnx_output),
        "--image_size", "224",
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        raise RuntimeError(f"export_model.py failed with exit code {result.returncode}")
    
    print(f"\nSuccessfully exported best SAHL model to: {onnx_output}")
    print(f"Model metrics: F1={best_f1:.4f}, MAE={best_mae:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select and export the best 2.5x SAHL model from variance study."
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    export_best_sahl_model()


if __name__ == "__main__":
    main()
