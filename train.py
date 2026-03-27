import argparse
import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from dataset import create_dataloaders, set_seed


def build_model(freeze_early_layers: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_early_layers:
        modules_to_freeze = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )

    return model


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if predictions.numel() == 0:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}

    predictions = predictions.view(-1)
    targets = targets.view(-1)
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    ss_res = torch.sum((targets - predictions) ** 2)
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    r2 = 0.0 if ss_tot.item() <= 1e-12 else (1.0 - (ss_res / ss_tot)).item()

    return {"mse": mse, "mae": mae, "r2": r2}


def evaluate(model: nn.Module, data_loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    avg_loss = running_loss / max(total, 1)
    predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.empty(0, 1)
    targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, 1)
    metrics = compute_regression_metrics(predictions, targets)
    metrics["loss"] = avg_loss
    return metrics


@dataclass
class EarlyStopping:
    patience: int = 7
    min_delta: float = 1e-4
    best_score: float = float("inf")
    epochs_without_improvement: int = 0

    def step(self, current_score: float) -> bool:
        if current_score < (self.best_score - self.min_delta):
            self.best_score = current_score
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def build_loss(loss_type: str, huber_beta: float) -> nn.Module:
    if loss_type == "smoothl1":
        return nn.SmoothL1Loss(beta=huber_beta)
    if loss_type == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, test_loader, split_counts = create_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
    )

    model = build_model(freeze_early_layers=True).to(device)
    criterion = build_loss(args.loss_type, args.huber_beta)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_delta)

    best_val_mse = float("inf")
    best_val_mae = float("inf")
    best_state = None

    print(f"Device: {device}")
    print(f"Split counts: {split_counts}")

    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        n_train = 0
        train_predictions = []
        train_targets = []

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)
        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_train_loss += loss.item() * batch_size
            n_train += batch_size
            train_predictions.append(outputs.detach().cpu())
            train_targets.append(labels.detach().cpu())

            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_train_loss / max(n_train, 1)
        train_pred_tensor = torch.cat(train_predictions, dim=0) if train_predictions else torch.empty(0, 1)
        train_target_tensor = torch.cat(train_targets, dim=0) if train_targets else torch.empty(0, 1)
        train_metrics = compute_regression_metrics(train_pred_tensor, train_target_tensor)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train MSE: {train_metrics['mse']:.4f} | "
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Train R2: {train_metrics['r2']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_val_mse = val_metrics["mse"]
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                "model_state_dict": best_state,
                "image_size": args.image_size,
                "best_val_mse": best_val_mse,
                "best_val_mae": best_val_mae,
                "best_epoch": epoch + 1,
                "loss_type": args.loss_type,
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"Saved improved checkpoint to: {args.checkpoint_path}")

        if early_stopping.step(val_metrics["mae"]):
            print(f"Early stopping triggered at epoch {epoch + 1} (best val MAE: {early_stopping.best_score:.4f}).")
            break

    if best_state is None:
        raise RuntimeError("Training completed but no best model state was captured.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"Test MSE: {test_metrics['mse']:.4f} | "
        f"Test MAE: {test_metrics['mae']:.4f} | "
        f"Test R2: {test_metrics['r2']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 EL defect regression model with transfer learning.")
    parser.add_argument("--csv_path", type=str, default="labels.csv")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss_type", type=str, default="smoothl1", choices=["smoothl1", "mse"])
    parser.add_argument("--huber_beta", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stopping_patience", type=int, default=7)
    parser.add_argument("--early_stopping_delta", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
