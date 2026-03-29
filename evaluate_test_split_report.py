import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch

from dataset import build_transforms, load_csv_samples, stratified_split, ELDataset
from inference_mqtt_mock import infer_severity_score
from train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-image prediction error report on the held-out test split."
    )
    parser.add_argument("--csv_path", type=str, default="labels.csv")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_csv", type=str, default="test_split_report.csv")

    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--onnx_model", type=str, default="")
    parser.add_argument("--device", type=str, default="")

    return parser.parse_args()


def resolve_test_items(csv_path: Path, data_root: Path, seed: int) -> List[Tuple[Path, float]]:
    all_samples = load_csv_samples(csv_path, data_root)
    split_items = stratified_split(all_samples, 0.70, 0.15, 0.15, seed)
    return split_items.test


def predict_with_checkpoint(
    checkpoint_path: Path,
    items: Sequence[Tuple[Path, float]],
    image_size: int,
    device: torch.device,
) -> List[float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(freeze_early_layers=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_tfms, _ = build_transforms(image_size=image_size)
    test_ds = ELDataset(items, eval_tfms)

    predictions: List[float] = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            image_tensor, _ = test_ds[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)
            output = model(image_tensor)
            predictions.append(float(output[0, 0].item()))

    return predictions


def predict_with_onnx(onnx_path: Path, items: Sequence[Tuple[Path, float]], image_size: int) -> List[float]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    predictions: List[float] = []
    for image_path, _ in items:
        pred = infer_severity_score(
            onnx_model_path=str(onnx_path),
            image_path=str(image_path),
            image_size=image_size,
            session=session,
        )
        predictions.append(float(pred))

    return predictions


def write_report(output_csv: Path, items: Sequence[Tuple[Path, float]], predictions: Sequence[float]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    abs_errors = []
    squared_errors = []

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["image_path", "target", "prediction", "abs_error", "squared_error"])

        for (image_path, target), pred in zip(items, predictions):
            abs_err = abs(pred - target)
            sq_err = (pred - target) ** 2
            abs_errors.append(abs_err)
            squared_errors.append(sq_err)
            writer.writerow([
                str(image_path),
                f"{target:.6f}",
                f"{pred:.6f}",
                f"{abs_err:.6f}",
                f"{sq_err:.6f}",
            ])

    mae = float(np.mean(abs_errors)) if abs_errors else float("nan")
    mse = float(np.mean(squared_errors)) if squared_errors else float("nan")

    print(f"Report saved: {output_csv}")
    print(f"Rows: {len(abs_errors)}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    data_root = Path(args.data_root)
    output_csv = Path(args.output_csv)

    if bool(args.checkpoint) == bool(args.onnx_model):
        raise ValueError("Provide exactly one of --checkpoint or --onnx_model.")

    items = resolve_test_items(csv_path=csv_path, data_root=data_root, seed=args.seed)

    if args.checkpoint:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        predictions = predict_with_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            items=items,
            image_size=args.image_size,
            device=device,
        )
    else:
        predictions = predict_with_onnx(
            onnx_path=Path(args.onnx_model),
            items=items,
            image_size=args.image_size,
        )

    write_report(output_csv=output_csv, items=items, predictions=predictions)


if __name__ == "__main__":
    main()
