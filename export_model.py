import argparse
from pathlib import Path

import onnx
import torch

from train import build_model


def export_to_onnx(
    checkpoint_path: str,
    onnx_output: str,
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    model = build_model(freeze_early_layers=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    onnx_path = Path(onnx_output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["severity_score"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "severity_score": {0: "batch_size"},
        },
    )

    loaded = onnx.load(str(onnx_path))
    onnx.checker.check_model(loaded)

    print(f"ONNX export complete: {onnx_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained EL classifier checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--onnx_output", type=str, default="best_model.onnx")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        onnx_output=args.onnx_output,
        image_size=args.image_size,
        opset_version=args.opset,
    )
