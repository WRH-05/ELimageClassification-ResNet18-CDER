"""
Orchestrate multi-seed variance studies and SAHL weight ablation.

Runs train → export → report pipeline for multiple configurations.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    seed: int
    loss_type: str
    loss_weight_multiplier: float
    loss_weight_threshold: float = 0.66
    critical_recall_threshold: float = 0.6
    critical_target_threshold: float = 0.8
    precision_floor: float = 0.70
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    csv_path: str = "labels.csv"
    data_root: str = "."
    image_size: int = 224
    device: str = "cuda"

    def get_checkpoint_name(self) -> str:
        """Generate unique checkpoint filename for this configuration."""
        return (
            f"best_model_seed{self.seed}_"
            f"{self.loss_type}_w{self.loss_weight_multiplier:.1f}.pth"
        )

    def get_onnx_name(self) -> str:
        """Generate unique ONNX filename for this configuration."""
        return (
            f"best_model_seed{self.seed}_"
            f"{self.loss_type}_w{self.loss_weight_multiplier:.1f}.onnx"
        )

    def get_report_name(self) -> str:
        """Generate unique report CSV filename for this configuration."""
        return (
            f"test_split_report_seed{self.seed}_"
            f"{self.loss_type}_w{self.loss_weight_multiplier:.1f}.csv"
        )


def invoke_python_step(
    step_name: str,
    script_path: str,
    args: List[str],
) -> None:
    """Run a Python script as a subprocess and check for errors."""
    python_exe = Path(".venv") / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found at {python_exe}")

    cmd = [str(python_exe), script_path] + args
    print(f"\n{'='*70}")
    print(f"  {step_name}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def run_single_experiment(config: ExperimentConfig) -> None:
    """Execute train → export → report for a single configuration."""
    checkpoint_path = Path("models/pth") / config.get_checkpoint_name()
    onnx_path = Path("models/onnx") / config.get_onnx_name()
    report_path = Path("testCsv") / config.get_report_name()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning experiment: {config.loss_type} @ {config.loss_weight_multiplier}x, seed={config.seed}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  ONNX: {onnx_path}")
    print(f"  Report: {report_path}")

    if report_path.exists():
        print("  Resume mode: report already exists, skipping this experiment.")
        return

    # Step 1: Train
    if not checkpoint_path.exists():
        train_args = [
            "--csv_path", config.csv_path,
            "--data_root", config.data_root,
            "--epochs", str(config.epochs),
            "--batch_size", str(config.batch_size),
            "--learning_rate", str(config.learning_rate),
            "--seed", str(config.seed),
            "--checkpoint_path", str(checkpoint_path),
            "--loss_type", config.loss_type,
            "--loss_weight_threshold", str(config.loss_weight_threshold),
            "--loss_weight_multiplier", str(config.loss_weight_multiplier),
            "--critical_recall_threshold", str(config.critical_recall_threshold),
            "--critical_target_threshold", str(config.critical_target_threshold),
            "--precision_floor", str(config.precision_floor),
            "--image_size", str(config.image_size),
            "--device", config.device,
        ]
        invoke_python_step("Train", "train.py", train_args)
    else:
        print("  Resume mode: checkpoint already exists, skipping train step.")

    # Step 2: Export to ONNX
    if not onnx_path.exists():
        export_args = [
            "--checkpoint", str(checkpoint_path),
            "--onnx_output", str(onnx_path),
            "--image_size", str(config.image_size),
        ]
        invoke_python_step("Export ONNX", "export_model.py", export_args)
    else:
        print("  Resume mode: ONNX already exists, skipping export step.")

    # Step 3: Generate test report
    report_args = [
        "--csv_path", config.csv_path,
        "--data_root", config.data_root,
        "--seed", str(config.seed),
        "--image_size", str(config.image_size),
        "--onnx_model", str(onnx_path),
        "--output_csv", str(report_path),
    ]
    invoke_python_step("Generate Test Report", "evaluate_test_split_report.py", report_args)


def run_variance_study(args: argparse.Namespace) -> None:
    """
    Run the statistical variance study (multi-seed).
    
    V1 (MSE baseline) and V3.1 SAHL (weighted_l1 @ 2.5x) across seeds 42, 123, 2026.
    """
    seeds = [42, 123, 2026]
    configs: List[ExperimentConfig] = []

    # V1: MSE baseline
    for seed in seeds:
        configs.append(ExperimentConfig(
            seed=seed,
            loss_type="mse",
            loss_weight_multiplier=1.0,  # MSE doesn't use the multiplier, but set for consistency
            precision_floor=args.precision_floor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            csv_path=args.csv_path,
            data_root=args.data_root,
            image_size=args.image_size,
            device=args.device,
        ))

    # V3.1: weighted_l1 @ 2.5x
    for seed in seeds:
        configs.append(ExperimentConfig(
            seed=seed,
            loss_type="weighted_l1",
            loss_weight_multiplier=2.5,
            precision_floor=args.precision_floor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            csv_path=args.csv_path,
            data_root=args.data_root,
            image_size=args.image_size,
            device=args.device,
        ))

    for config in configs:
        run_single_experiment(config)

    print("\n" + "="*70)
    print("  Variance study complete!")
    print("  Next: run aggregate_results.py --mode variance")
    print("="*70 + "\n")


def run_ablation_study(args: argparse.Namespace) -> None:
    """
    Run the SAHL weight ablation (single seed, varying weight).
    
    weighted_l1 with weights 1.0x, 1.5x, 2.5x, 5.0x, fixed seed=42.
    """
    seed = 42
    weights = [1.0, 1.5, 2.5, 5.0]
    configs: List[ExperimentConfig] = []

    for weight in weights:
        configs.append(ExperimentConfig(
            seed=seed,
            loss_type="weighted_l1",
            loss_weight_multiplier=weight,
            precision_floor=args.precision_floor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            csv_path=args.csv_path,
            data_root=args.data_root,
            image_size=args.image_size,
            device=args.device,
        ))

    for config in configs:
        run_single_experiment(config)

    print("\n" + "="*70)
    print("  Ablation study complete!")
    print("  Next: run aggregate_results.py --mode ablation")
    print("="*70 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run variance studies and ablation sweeps for multi-seed experiments."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="variance",
        choices=["variance", "ablation"],
        help="Which study to run: 'variance' (multi-seed) or 'ablation' (weight sweep).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="labels.csv",
        help="Path to the labels CSV.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Data root directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training.",
    )
    parser.add_argument(
        "--precision_floor",
        type=float,
        default=0.70,
        help="Minimum precision required before checkpoint save.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        if args.mode == "variance":
            run_variance_study(args)
        elif args.mode == "ablation":
            run_ablation_study(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
