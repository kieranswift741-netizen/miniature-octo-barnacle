"""
PyTorch Dataset for quantum battery time-series data.

Implements sliding-window mechanism: given T_win consecutive density matrix
feature vectors, predict the density matrix at the next time step.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class QuantumBatteryDataset(Dataset):
    """
    Sliding-window PyTorch Dataset for quantum battery evolution data.

    Each sample:
      X: [window_size, feature_dim] — past density matrix snapshots
      y: [output_dim] — density matrix features at the next time step
         (output_dim = 2 * d²: real parts then imaginary parts)

    Features are standardized to zero mean and unit variance using
    statistics computed from the training split only.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 20,
        train: bool = True,
        train_split: float = 0.8,
        seed: int = 42,
        feature_columns: Optional[list] = None,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            data_dir: Directory containing .npz trajectory files and metadata.json.
            window_size: Number of past time steps used as input (T_win).
            train: If True, return training split; else validation split.
            train_split: Fraction of trajectories for training.
            seed: Random seed for reproducible splits.
            feature_columns: Optional subset of feature indices (default: all).
            feature_mean: Pre-computed feature mean for standardization.
            feature_std: Pre-computed feature std for standardization.
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.train = train
        self.train_split = train_split
        self.seed = seed

        # Load metadata
        meta_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata not found at {meta_path}. Run generate.py first."
            )

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        # Split trajectories into train/val
        rng = np.random.RandomState(seed)
        n_traj = len(self.metadata)
        indices = rng.permutation(n_traj)
        split_point = int(n_traj * train_split)

        if train:
            self.traj_indices = indices[:split_point]
        else:
            self.traj_indices = indices[split_point:]

        # Build sample index: list of (traj_idx, start_step) pairs
        self.samples = []
        for idx in self.traj_indices:
            meta = self.metadata[idx]
            n_steps = meta["num_time_steps"]
            n_samples = n_steps - self.window_size
            for start in range(n_samples):
                self.samples.append((idx, start))

        # Determine feature dimension from first trajectory
        first_file = self.metadata[0]["file"]
        data = np.load(first_file)
        full_feature_dim = data["features"].shape[1]
        self.feature_dim = full_feature_dim
        self.output_dim = full_feature_dim

        if feature_columns is not None:
            self.feature_columns = feature_columns
            self.feature_dim = len(feature_columns)
            self.output_dim = len(feature_columns)
        else:
            self.feature_columns = list(range(full_feature_dim))

        # Feature normalization
        if feature_mean is not None and feature_std is not None:
            self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
            self.feature_std = np.asarray(feature_std, dtype=np.float32)
        else:
            self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
            self.feature_std = np.ones(self.feature_dim, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_idx, start_step = self.samples[idx]
        meta = self.metadata[traj_idx]

        data = np.load(meta["file"])

        # Extract window
        end_step = start_step + self.window_size
        features = data["features"][start_step:end_step]
        features = features[:, self.feature_columns]

        # Target: density matrix features at the next time step
        target = data["features"][end_step]
        target = target[self.feature_columns]

        # Standardize
        features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        target = (target - self.feature_mean) / (self.feature_std + 1e-8)

        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

        # Extract physical parameters for conditional input
        spectral = meta.get("spectral", {})
        params = meta.get("params", {})
        omega = spectral.get("Omega", params.get("coupling_strength", 0.1))
        gamma = spectral.get("gamma", params.get("spectral_width", 0.1))
        tau_m = 1.0 / gamma if gamma > 0 else 20.0  # memory time
        non_markov = omega / gamma if gamma > 0 else 0.0  # non-Markovianity measure

        # Normalize parameters to [0, 1] range (based on expected ranges)
        omega_norm = min(omega / 0.5, 1.0)  # Ω ∈ [0, 0.5]
        gamma_norm = min(gamma / 2.0, 1.0)  # γ ∈ [0, 2.0]
        tau_m_norm = min(tau_m / 50.0, 1.0)  # τ_m ∈ [0, 50]
        non_markov_norm = min(non_markov / 10.0, 1.0)  # Ω/γ ∈ [0, 10]

        cond = torch.tensor([omega_norm, gamma_norm, tau_m_norm, non_markov_norm], dtype=torch.float32)

        return X, y, cond

    def get_full_trajectory(self, traj_idx: int) -> dict:
        """Load a full trajectory for evaluation/plotting."""
        meta = self.metadata[traj_idx]
        data = np.load(meta["file"])
        return {
            "times": data["times"],
            "features": data["features"],
            "energies": data["energies"],
            "ergotropies": data["ergotropies"],
            "params": meta["params"],
            "spectral": meta["spectral"],
            "file": meta["file"],
        }

    def get_metadata(self) -> list:
        """Return list of trajectory metadata dicts."""
        return self.metadata

    @property
    def feature_dim_value(self) -> int:
        """Return the input feature dimension."""
        return self.feature_dim

    @property
    def output_dim_value(self) -> int:
        """Return the output dimension (= input feature dimension)."""
        return self.output_dim

    def __repr__(self) -> str:
        split_name = "train" if self.train else "val"
        return (
            f"QuantumBatteryDataset(split={split_name}, "
            f"n_samples={len(self)}, window={self.window_size}, "
            f"feature_dim={self.feature_dim})"
        )


def create_dataloaders(
    data_dir: str,
    window_size: int = 20,
    batch_size: int = 64,
    train_split: float = 0.8,
    seed: int = 42,
):
    """
    Create train and validation DataLoaders with feature standardization.

    Normalization statistics are computed from the training split only
    and applied to both train and validation sets.

    Returns:
        (train_loader, val_loader, feature_dim)
    """
    # First pass: create temporary training dataset to compute statistics
    temp_train = QuantumBatteryDataset(
        data_dir=data_dir,
        window_size=window_size,
        train=True,
        train_split=train_split,
        seed=seed,
    )

    # Compute feature statistics from training data
    all_features = []
    for idx in range(len(temp_train)):
        sample = temp_train[idx]
        # Handle both old (X, y) and new (X, y, cond) formats
        X = sample[0]
        all_features.append(X.numpy())
    all_features = np.concatenate(all_features, axis=0)  # (n_windows, feat_dim)
    feature_mean = all_features.mean(axis=0)
    feature_std = all_features.std(axis=0)
    # Don't scale features that have near-zero variance
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    # Create final datasets with normalization
    train_dataset = QuantumBatteryDataset(
        data_dir=data_dir,
        window_size=window_size,
        train=True,
        train_split=train_split,
        seed=seed,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    val_dataset = QuantumBatteryDataset(
        data_dir=data_dir,
        window_size=window_size,
        train=False,
        train_split=train_split,
        seed=seed,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, train_dataset.feature_dim_value


if __name__ == "__main__":
    import tempfile

    tmpdir = tempfile.mkdtemp()

    dummy_features = np.random.randn(100, 8)
    dummy_energies = np.random.randn(100)
    dummy_ergotropies = np.random.randn(100)
    np.savez_compressed(
        os.path.join(tmpdir, "traj_0000.npz"),
        times=np.arange(100),
        features=dummy_features,
        energies=dummy_energies,
        ergotropies=dummy_ergotropies,
    )

    metadata = [
        {
            "file": os.path.join(tmpdir, "traj_0000.npz"),
            "index": 0,
            "num_time_steps": 100,
            "feature_dim": 8,
            "params": {"coupling_strength": 0.1, "spectral_width": 0.5},
            "spectral": {},
        }
    ]
    with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    ds = QuantumBatteryDataset(tmpdir, window_size=20, train=True)
    print(f"Dataset: {ds}")
    print(f"Num samples: {len(ds)}")
    X, y = ds[0]
    print(f"Sample X shape: {X.shape}, y shape: {y.shape}")
