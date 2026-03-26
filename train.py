import argparse
import copy
from typing import Tuple

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


def evaluate(model: nn.Module, data_loader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    running_mae = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            mae = torch.abs(outputs - labels).mean()

            running_loss += loss.item() * images.size(0)
            running_mae += mae.item() * images.size(0)
            total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    avg_mae = running_mae / max(total, 1)
    return avg_loss, avg_mae


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
    criterion = nn.MSELoss()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_mse = float("inf")
    best_state = None

    print(f"Device: {device}")
    print(f"Split counts: {split_counts}")

    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        n_train = 0

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

            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_train_loss / max(n_train, 1)
        val_mse, val_mae = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                "model_state_dict": best_state,
                "image_size": args.image_size,
                "best_val_mse": best_val_mse,
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"Saved improved checkpoint to: {args.checkpoint_path}")

    if best_state is None:
        raise RuntimeError("Training completed but no best model state was captured.")

    model.load_state_dict(best_state)
    test_mse, test_mae = evaluate(model, test_loader, criterion, device)
    print(
        f"Test MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f}"
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
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
