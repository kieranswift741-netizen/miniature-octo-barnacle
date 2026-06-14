"""
Training loop for the Quantum Battery LSTM model.

Includes:
  - Config-driven hyperparameters
  - Early stopping on validation loss
  - Learning rate scheduler (ReduceLROnPlateau)
  - Model checkpointing (best + periodic)
  - Loss component logging for physics constraints
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import create_dataloaders, QuantumBatteryDataset
from src.model.lstm import build_model
from src.model.loss import PhysicsInformedLoss


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_d_from_datadir(data_dir: str) -> int:
    """Infer Hilbert space dimension d from the data."""
    ds = QuantumBatteryDataset(data_dir, window_size=20, train=True)
    # feature_dim = 2 * d²  →  d = sqrt(feature_dim / 2)
    half = ds.feature_dim_value // 2
    return int(half ** 0.5)


class EarlyStopping:
    """Early stopping with patience and model checkpointing."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # no improvement


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: PhysicsInformedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    feat_mean_t: torch.Tensor = None,
    feat_std_t: torch.Tensor = None,
) -> dict:
    """Run one training epoch. Denormalizes predictions before physics constraints."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    use_denorm = feat_mean_t is not None and feat_std_t is not None
    if use_denorm:
        feat_mean_t = feat_mean_t.to(device)
        feat_std_t = feat_std_t.to(device)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        # Support both (X, y) and (X, y, cond) formats
        if len(batch) == 3:
            X, y, cond = batch
            X, y, cond = X.to(device), y.to(device), cond.to(device)
        else:
            X, y = batch
            X, y = X.to(device), y.to(device)
            cond = None

        optimizer.zero_grad()
        y_pred = model(X, cond=cond)
        if use_denorm:
            y_pred_raw = y_pred * feat_std_t + feat_mean_t
            y_raw = y * feat_std_t + feat_mean_t
            loss = criterion(y_pred_raw, y_raw)
        else:
            loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / n_batches
    components = criterion.get_component_losses()

    return {"train_loss": avg_loss, **{f"train_{k}": v for k, v in components.items()}}


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    feat_mean_t: torch.Tensor = None,
    feat_std_t: torch.Tensor = None,
) -> dict:
    """Run one validation epoch. Denormalizes predictions before physics constraints."""
    model.eval()
    total_loss = 0.0

    use_denorm = feat_mean_t is not None and feat_std_t is not None
    if use_denorm:
        feat_mean_t = feat_mean_t.to(device)
        feat_std_t = feat_std_t.to(device)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False)
    for batch in pbar:
        # Support both (X, y) and (X, y, cond) formats
        if len(batch) == 3:
            X, y, cond = batch
            X, y, cond = X.to(device), y.to(device), cond.to(device)
        else:
            X, y = batch
            X, y = X.to(device), y.to(device)
            cond = None

        y_pred = model(X, cond=cond)
        if use_denorm:
            y_pred_raw = y_pred * feat_std_t + feat_mean_t
            y_raw = y * feat_std_t + feat_mean_t
            loss = criterion(y_pred_raw, y_raw)
        else:
            loss = criterion(y_pred, y)
        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(loader)
    components = criterion.get_component_losses()

    return {"val_loss": avg_loss, **{f"val_{k}": v for k, v in components.items()}}


def train(
    config_path: str = "config.yaml",
    data_dir: str = None,
    model_dir: str = None,
    log_dir: str = None,
    device_str: str = None,
):
    """Main training function."""
    config = load_config(config_path)

    cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    data_cfg = config["data"]

    # Resolve paths
    if data_dir is None:
        data_dir = data_cfg.get("output_dir", "./data/generated")
    if model_dir is None:
        model_dir = config.get("output", {}).get("model_dir", "./models")
    if log_dir is None:
        log_dir = config.get("output", {}).get("log_dir", "./logs")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    window_size = data_cfg.get("window_size", 20)
    batch_size = cfg.get("batch_size", 64)
    train_split = data_cfg.get("train_split", 0.8)

    train_loader, val_loader, feature_dim = create_dataloaders(
        data_dir=data_dir,
        window_size=window_size,
        batch_size=batch_size,
        train_split=train_split,
        seed=seed,
    )

    # Model
    d = compute_d_from_datadir(data_dir)
    output_dim = feature_dim  # predicting full density matrix

    # Adaptive architecture: scale with feature_dim to avoid bottleneck
    # Config can override with explicit values; otherwise auto-scale
    if model_cfg.get("lstm_layers") is not None:
        lstm_layers = model_cfg["lstm_layers"]
    else:
        h1 = max(64, min(256, feature_dim))
        h2 = max(32, h1 // 2)
        lstm_layers = [h1, h2]
    if model_cfg.get("dense_units") is not None:
        dense_units = model_cfg["dense_units"]
    else:
        dense_units = max(16, lstm_layers[-1] // 2)
    dropout = model_cfg.get("dropout", 0.2)

    model = build_model(
        input_dim=feature_dim,
        output_dim=output_dim,
        lstm_layers=lstm_layers,
        dense_units=dense_units,
        dropout=dropout,
        cond_dim=4,  # Ω, γ, τ_m, Ω/γ
    ).to(device)

    # Normalization stats (saved for evaluation, used for denorm in physics loss)
    feature_mean = train_loader.dataset.feature_mean
    feature_std = train_loader.dataset.feature_std
    feat_mean_t = torch.tensor(feature_mean, dtype=torch.float32)
    feat_std_t = torch.tensor(feature_std, dtype=torch.float32)

    print(f"Model: {model}")
    print(f"Hilbert space dimension d = {d}")
    print(f"Feature dim = {feature_dim}, Output dim = {output_dim}")
    print(f"LSTM layers = {lstm_layers}, Dense units = {dense_units}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Loss
    criterion = PhysicsInformedLoss(
        lambda_trace=loss_cfg.get("lambda_trace", 1.0),
        lambda_herm=loss_cfg.get("lambda_herm", 1.0),
        lambda_pos=loss_cfg.get("lambda_pos", 0.1),
        use_physics_constraints=True,
        d=d,
    )

    # Optimizer
    lr = cfg.get("learning_rate", 0.001)
    weight_decay = cfg.get("weight_decay", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler_cfg = cfg.get("lr_scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 10),
    )

    # Early stopping
    patience = cfg.get("early_stop_patience", 30)
    early_stopper = EarlyStopping(patience=patience)

    # Training loop
    epochs = cfg.get("epochs", 200)
    history = []
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs (early stop patience={patience})")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
            feat_mean_t, feat_std_t,
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, epochs,
            feat_mean_t, feat_std_t,
        )

        # Combine
        metrics = {**train_metrics, **val_metrics, "epoch": epoch}
        history.append(metrics)

        # Scheduler step
        scheduler.step(val_metrics["val_loss"])

        # Early stopping
        improved = early_stopper(val_metrics["val_loss"])

        # Print summary
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_metrics['train_loss']:.6f} | "
            f"Val: {val_metrics['val_loss']:.6f} | "
            f"LR: {lr_now:.2e} | "
            f"Best: {early_stopper.best_loss:.6f}"
        )

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint_path = os.path.join(model_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                    "feature_dim": feature_dim,
                    "output_dim": output_dim,
                    "d": d,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "lstm_layers": lstm_layers,
                    "dense_units": dense_units,
                    "cond_dim": model.cond_dim,
                },
                checkpoint_path,
            )
            print(f"  [OK] Saved best model to {checkpoint_path}")

        if early_stopper.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*60}")

    # Save final model
    final_path = os.path.join(model_dir, "final_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
            "config": config,
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "d": d,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "lstm_layers": lstm_layers,
            "dense_units": dense_units,
        },
        final_path,
    )
    print(f"Saved final model to {final_path}")

    # Save training history
    history_path = os.path.join(log_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train the Quantum Battery LSTM model."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to generated data directory."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cpu, cuda, mps)."
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_dir=args.data,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
