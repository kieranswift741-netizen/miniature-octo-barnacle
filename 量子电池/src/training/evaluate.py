"""
Evaluation metrics for the Quantum Battery LSTM model.

Computes:
  - Energy E(t) = Tr[ρ(t) H_S]
  - Ergotropy E(ρ) = max extractable work
  - Efficiency η = E / E_input
  - Average power P(t) = E(t) / t
  - Prediction error (MSE, MAE on energy)
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import QuantumBatteryDataset
from src.model.lstm import build_model
from src.model.loss import reconstruct_density_matrix


def load_trained_model(
    checkpoint_path: str,
    device: torch.device = None,
) -> tuple:
    """
    Load a trained model from checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    feature_dim = checkpoint.get("feature_dim", 8)
    output_dim = checkpoint.get("output_dim", 8)

    # Use saved architecture from checkpoint (set by train.py adaptive scaling).
    # Fall back to config values, then to adaptive defaults.
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})

    lstm_layers = checkpoint.get("lstm_layers")
    if lstm_layers is None:
        lstm_layers = model_cfg.get("lstm_layers")
    if lstm_layers is None:
        h1 = max(64, min(256, feature_dim))
        h2 = max(32, h1 // 2)
        lstm_layers = [h1, h2]

    dense_units = checkpoint.get("dense_units")
    if dense_units is None:
        dense_units = model_cfg.get("dense_units")
    if dense_units is None:
        dense_units = max(16, lstm_layers[-1] // 2)

    # Read cond_dim from checkpoint (0 for old models without conditional params)
    cond_dim = checkpoint.get("cond_dim", 0)

    model = build_model(
        input_dim=feature_dim,
        output_dim=output_dim,
        lstm_layers=lstm_layers,
        dense_units=dense_units,
        dropout=model_cfg.get("dropout", 0.2),
        cond_dim=cond_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def reconstruct_hamiltonian(
    omega_0: float = 1.0,
    num_tls: int = 1,
    d: int = 2,
) -> torch.Tensor:
    """
    Build the bare system Hamiltonian matrix in energy basis.

    For N TLS: H_S = Σ (ω₀/2) σ_z^(i), diagonal in computational basis.

    Returns:
        Complex tensor of shape (d, d).
    """
    # Build diagonal Hamiltonian
    import numpy as np
    from itertools import product

    states = list(product([0, 1], repeat=num_tls))
    energies = []
    for state in states:
        e = 0.0
        for s in state:
            e += omega_0 * (0.5 if s == 0 else -0.5)
        energies.append(e)

    H_diag = np.diag(energies)
    return torch.tensor(H_diag, dtype=torch.complex64)


def compute_energy_from_rho(
    rho_pred: torch.Tensor,
    H_S: torch.Tensor,
) -> np.ndarray:
    """
    Compute energy E = Tr(ρ H_S) from predicted density matrices.

    Args:
        rho_pred: Complex density matrices (n_steps, d, d).
        H_S: Hamiltonian matrix (d, d).

    Returns:
        Energy values (n_steps,).
    """
    energies = []
    for rho in rho_pred:
        e = torch.real(torch.trace(torch.matmul(rho, H_S)))
        energies.append(e.item())
    return np.array(energies)


def compute_ergotropy_from_rho(
    rho: torch.Tensor,
    H_S: torch.Tensor,
) -> float:
    """
    Compute ergotropy from a predicted density matrix.

    E = Tr(ρ H_S) - Tr(σ_ρ H_S)
    where σ_ρ is the passive state (populations decreasing with energy).

    Args:
        rho: Complex density matrix (d, d).
        H_S: Hamiltonian matrix (d, d).

    Returns:
        Ergotropy value.
    """
    # Diagonalize H_S
    H_np = H_S.real.numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    idx = np.argsort(eigenvalues)  # increasing energy
    eigenvalues_sorted = eigenvalues[idx]

    # Populations in energy basis
    rho_np = rho.detach().numpy()
    rho_energy_basis = eigenvectors.conj().T @ rho_np @ eigenvectors
    populations = np.real(np.diag(rho_energy_basis))[idx]

    # Internal energy
    internal_energy = np.real(np.trace(rho_np @ H_np))

    # Passive state energy: sort populations decreasing, multiply by increasing energies
    rho_eigenvalues = np.sort(np.real(np.linalg.eigvals(rho_np)))[::-1]
    passive_energy = np.sum(rho_eigenvalues * eigenvalues_sorted)

    return max(0.0, internal_energy - passive_energy)


def evaluate_model(
    model: torch.nn.Module,
    dataset: QuantumBatteryDataset,
    H_S: torch.Tensor,
    device: torch.device,
    n_trajectories: int = 5,
    feature_mean: np.ndarray = None,
    feature_std: np.ndarray = None,
) -> dict:
    """
    Evaluate model on multiple full trajectories.

    For each trajectory, run autoregressive prediction from the initial window
    and compare predicted vs true energy evolution.

    Args:
        model: Trained LSTM model.
        dataset: QuantumBatteryDataset (should have matching normalization).
        H_S: Bare system Hamiltonian matrix.
        device: Torch device.
        n_trajectories: Number of trajectories to evaluate.
        feature_mean: Feature mean for denormalization (if model trained with normalization).
        feature_std: Feature std for denormalization.

    Returns:
        dict with per-trajectory metrics and aggregate summary.
    """
    model.eval()
    results = []

    use_norm = feature_mean is not None and feature_std is not None
    if use_norm:
        fm = np.asarray(feature_mean, dtype=np.float32)
        fs = np.asarray(feature_std, dtype=np.float32)
        fm_t = torch.tensor(fm, dtype=torch.float32).to(device)
        fs_t = torch.tensor(fs, dtype=torch.float32).to(device)

    val_indices = dataset.traj_indices[:n_trajectories]

    for traj_idx in val_indices:
        traj = dataset.get_full_trajectory(traj_idx)
        features_raw = traj["features"]
        true_energies = traj["energies"]

        window_size = dataset.window_size
        n_total = len(features_raw)
        n_pred_steps = n_total - window_size

        # Normalize features if needed
        if use_norm:
            features = (features_raw - fm) / (fs + 1e-8)
        else:
            features = features_raw

        # Initial window
        initial_window = torch.tensor(
            features[:window_size], dtype=torch.float32
        ).unsqueeze(0).to(device)  # (1, window, feat_dim)

        # Extract conditional parameters for this trajectory
        spectral = traj.get("spectral", {})
        params = traj.get("params", {})
        omega = spectral.get("Omega", params.get("coupling_strength", 0.1))
        gamma = spectral.get("gamma", params.get("spectral_width", 0.1))
        tau_m = 1.0 / gamma if gamma > 0 else 20.0
        non_markov = omega / gamma if gamma > 0 else 0.0
        
        # Normalize parameters (same as in dataset.py)
        omega_norm = min(omega / 0.5, 1.0)
        gamma_norm = min(gamma / 2.0, 1.0)
        tau_m_norm = min(tau_m / 50.0, 1.0)
        non_markov_norm = min(non_markov / 10.0, 1.0)
        
        cond = torch.tensor(
            [[omega_norm, gamma_norm, tau_m_norm, non_markov_norm]],
            dtype=torch.float32
        ).to(device)

        # Autoregressive prediction
        model.return_hidden = False
        predictions_rho = []
        window = initial_window.clone()

        with torch.no_grad():
            for step in range(n_pred_steps):
                pred_norm = model(window, cond=cond).cpu()  # (1, output_dim)

                # Denormalize if needed
                if use_norm:
                    pred_raw = pred_norm[0].numpy() * fs + fm
                else:
                    pred_raw = pred_norm[0].numpy()

                # Reshape to density matrix
                half = len(pred_raw) // 2
                d = int(half ** 0.5)
                rho_real = pred_raw[:half].reshape(d, d)
                rho_imag = pred_raw[half:].reshape(d, d)
                rho = torch.complex(
                    torch.tensor(rho_real), torch.tensor(rho_imag)
                )
                predictions_rho.append(rho)

                # Roll window: use TRUE features (teacher forcing)
                true_next = torch.tensor(
                    features[window_size + step], dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0)  # (1, 1, feat_dim)
                window = torch.cat([window[:, 1:, :], true_next.to(device)], dim=1)

        # Compute predicted energies
        predicted_energies = compute_energy_from_rho(
            torch.stack(predictions_rho), H_S
        )

        # Compute predicted ergotropies
        predicted_ergotropies = np.array([
            compute_ergotropy_from_rho(rho, H_S) for rho in predictions_rho
        ])

        # Metrics (compare bare energy)
        true_pred_range = true_energies[window_size:]
        mse = np.mean((predicted_energies - true_pred_range) ** 2)
        mae = np.mean(np.abs(predicted_energies - true_pred_range))

        # Efficiency = Charging Power: P = E_peak / t_peak
        # Find the first local maximum in the energy trajectory
        times = np.asarray(traj["times"])

        def find_first_peak(energies, times):
            """Find first local maximum (charging peak) in energy trajectory."""
            for i in range(1, len(energies) - 1):
                if energies[i] > energies[i - 1] and energies[i] >= energies[i + 1]:
                    return float(energies[i]), float(times[i])
            # Fallback: use global max
            idx = np.argmax(energies)
            return float(energies[idx]), float(times[idx])

        e_peak_true, t_peak_true = find_first_peak(true_energies, times)
        e_peak_pred, t_peak_pred = find_first_peak(predicted_energies, times[window_size:])

        # Power = peak energy / time to peak (charging rate)
        power_true = e_peak_true / t_peak_true if t_peak_true > 0 else 0.0
        power_pred = e_peak_pred / t_peak_pred if t_peak_pred > 0 else 0.0

        # Also store full efficiency arrays for plotting (shifted to [0,1])
        d = len(H_S)
        num_tls = int(round(np.log2(d))) if d > 1 else 1
        omega_0_val = float(abs(H_S[1, 1].real - H_S[0, 0].real))
        ground_energy = -num_tls * omega_0_val / 2.0
        e_span = num_tls * omega_0_val
        true_efficiency = (true_energies[window_size:] - ground_energy) / e_span
        pred_efficiency = (predicted_energies - ground_energy) / e_span

        results.append({
            "traj_idx": int(traj_idx),
            "params": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in traj["params"].items()},
            "spectral": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in traj["spectral"].items()},
            "energy_mse": float(mse),
            "energy_mae": float(mae),
            "true_energies": true_energies.astype(float).tolist(),
            "predicted_energies": predicted_energies.astype(float).tolist(),
            "true_ergotropies": np.asarray(traj["ergotropies"]).astype(float).tolist(),
            "predicted_ergotropies": predicted_ergotropies.astype(float).tolist(),
            "times": np.asarray(traj["times"]).astype(float).tolist(),
            "efficiency_true": true_efficiency.astype(float).tolist(),
            "efficiency_pred": pred_efficiency.astype(float).tolist(),
            "peak_efficiency_true": power_true,
            "peak_efficiency_pred": power_pred,
            "e_peak_true": e_peak_true,
            "t_peak_true": t_peak_true,
            "e_peak_pred": e_peak_pred,
            "t_peak_pred": t_peak_pred,
        })

    # Aggregate
    avg_mse = np.mean([r["energy_mse"] for r in results])
    avg_mae = np.mean([r["energy_mae"] for r in results])

    summary = {
        "n_trajectories": len(results),
        "avg_energy_mse": float(avg_mse),
        "avg_energy_mae": float(avg_mae),
        "results": results,
    }

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate trained LSTM on quantum battery trajectories."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./models/best_model.pt",
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--data", type=str, default="./data/generated",
        help="Path to generated data directory."
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=5,
        help="Number of validation trajectories to evaluate."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save evaluation results (JSON)."
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model, checkpoint = load_trained_model(args.checkpoint, device)

    config = checkpoint.get("config", {})
    physics_cfg = config.get("physics", {})
    omega_0 = physics_cfg.get("omega_0", 1.0)
    num_tls = physics_cfg.get("num_tls", 1)
    d = checkpoint.get("d", 2)

    H_S = reconstruct_hamiltonian(omega_0=omega_0, num_tls=num_tls, d=d)

    feature_mean = checkpoint.get("feature_mean", None)
    feature_std = checkpoint.get("feature_std", None)

    dataset = QuantumBatteryDataset(
        data_dir=args.data,
        window_size=config.get("data", {}).get("window_size", 20),
        train=False,
        train_split=config.get("data", {}).get("train_split", 0.8),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    print(f"Evaluating on {args.n_trajectories} trajectories...")
    summary = evaluate_model(
        model, dataset, H_S, device, n_trajectories=args.n_trajectories,
        feature_mean=feature_mean, feature_std=feature_std,
    )

    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Trajectories: {summary['n_trajectories']}")
    print(f"Avg Energy MSE: {summary['avg_energy_mse']:.6f}")
    print(f"Avg Energy MAE: {summary['avg_energy_mae']:.6f}")

    for i, r in enumerate(summary["results"]):
        print(f"\n  Traj {r['traj_idx']}: MSE={r['energy_mse']:.6f}, MAE={r['energy_mae']:.6f}")
        print(f"    params: Omega={r['params'].get('coupling_strength', '?')}, gamma={r['params'].get('spectral_width', '?')}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
