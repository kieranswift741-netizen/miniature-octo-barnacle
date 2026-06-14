#!/usr/bin/env python3
"""
Experiment B: Storage Lifetime (Shelf-Life) Prediction

After charging, the quantum battery undergoes free evolution (no driving)
coupled to the structured reservoir. The LSTM predicts how long the battery
retains >90% of its maximum energy for different environmental topologies.

Key question: How does the spectral width γ affect the storage lifetime?
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics.hamiltonian import (
    build_dicke_hamiltonian,
    coupling_operators,
    build_system_observables,
)
from src.physics.spectral_density import LorentzianSpectralDensity
from src.physics.dynamics import (
    run_heom_simulation,
    density_matrix_to_features,
    compute_energy,
    stored_energy,
)
from src.data.generate import load_config
from src.visualization.plot import plot_energy_evolution


def run_experiment_b(
    config_path: str = "config.yaml",
    output_dir: str = "./experiments/output/experiment_b",
    num_tls: int = None,
):
    """
    Run Experiment B: Storage Lifetime prediction.

    1. Generate post-charge free evolution data (no driving) for various γ
    2. Train LSTM on decay dynamics
    3. Extract shelf-life: time until energy drops below 90% of maximum
    4. Plot shelf-life vs γ

    NOTE: H_S = (ω₀/2)σ_z has eigenvalues {+0.5, -0.5} for ω₀=1.
    |0⟩ (basis index 0) is the EXCITED state (E=+0.5).
    |1⟩ (basis index 1) is the GROUND state (E=-0.5).
    For a charged battery, start from |0⟩ (excited state).
    """
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)

    # Override number of TLS if specified
    if num_tls is not None:
        config["physics"]["num_tls"] = num_tls

    physics_cfg = config["physics"]
    omega_0 = physics_cfg.get("omega_0", 1.0)

    # Set no driving for free evolution
    physics_cfg["drive_amplitude"] = 0.0

    # Dense gamma sweep — need many points to resolve shelf-life vs gamma
    config["data"]["param_grid"] = {
        "coupling_strength": [0.05, 0.1, 0.15, 0.2],
        "spectral_width": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
    }
    config["data"]["output_dir"] = os.path.join(output_dir, "data")
    config["heom"]["total_time"] = 40.0  # Longer decay observation
    config["heom"]["max_depth"] = 5      # Ensure HEOM convergence
    config["training"]["epochs"] = 100
    config["training"]["early_stop_patience"] = 20

    print("=" * 70)
    print("Experiment B: Storage Lifetime (Shelf-Life) Prediction")
    print("=" * 70)
    print(f"N_TLS = {physics_cfg['num_tls']}, omega_0 = {omega_0}")
    print(f"Drive OFF (free decay evolution)")
    print(f"gamma values: {config['data']['param_grid']['spectral_width']}")
    print(f"Initial state: |0> (excited, E=+{omega_0/2:.2f}, stored energy=1.0)")
    print()

    # Generate data (no driving, start from EXCITED state |0⟩)
    print("[1/4] Generating free-evolution trajectories...")

    import qutip as qt

    num_tls = physics_cfg["num_tls"]
    dims = [2] * num_tls
    # |0⟩ = basis(2, 0) is the EXCITED state of H_S = σ_z/2 (E=+0.5)
    # For N>1, |0...0⟩ has all TLS in excited state
    rho_0 = qt.ket2dm(qt.basis(2 ** num_tls, 0))
    rho_0.dims = [dims, dims]

    H_S = build_dicke_hamiltonian(
        num_tls=num_tls,
        omega_0=omega_0,
        drive_amplitude=0.0,
        t=0.0,
    )
    ops = coupling_operators(num_tls=num_tls, mode="collective")

    os.makedirs(config["data"]["output_dir"], exist_ok=True)
    metadata_list = []
    trajectory_idx = 0

    for i, gamma in enumerate(config["data"]["param_grid"]["spectral_width"]):
        for j, Omega in enumerate(config["data"]["param_grid"]["coupling_strength"]):
            sd = LorentzianSpectralDensity(
                coupling_strength=Omega,
                spectral_width=gamma,
                center_frequency=physics_cfg.get("center_frequency", 1.0),
            )

            tlist = np.arange(
                0, config["heom"]["total_time"], config["heom"]["time_step"]
            )

            try:
                result = run_heom_simulation(
                    H_S=H_S,
                    coupling_ops=ops,
                    spectral_density=sd,
                    rho_0=rho_0,
                    tlist=tlist,
                    max_depth=config["heom"]["max_depth"],
                )

                states = result["states"]
                H_S_bare = build_system_observables(num_tls, omega_0=omega_0)["H_S"]
                features = np.array([density_matrix_to_features(rho) for rho in states])
                energies = np.array([compute_energy(rho, H_S_bare) for rho in states])
                stored = np.array([stored_energy(rho, H_S_bare) for rho in states])
                ergotropies = np.array([0.0 for rho in states])  # placeholder

                # Validate physicality
                max_abs_feat = np.max(np.abs(features))
                if max_abs_feat > 10.0:
                    raise RuntimeError(
                        f"Unphysical trajectory: max|feature|={max_abs_feat:.1f}. "
                        f"HEOM may not be converged at max_depth={config['heom']['max_depth']}."
                    )

                filepath = os.path.join(config["data"]["output_dir"], f"traj_{trajectory_idx:04d}.npz")
                np.savez_compressed(
                    filepath,
                    times=tlist,
                    features=features,
                    energies=energies,
                    ergotropies=stored,  # Store "stored energy" for shelf-life analysis
                )

                metadata_list.append({
                    "file": filepath,
                    "index": trajectory_idx,
                    "params": {"coupling_strength": Omega, "spectral_width": gamma},
                    "spectral": sd.to_dict(),
                    "num_time_steps": len(tlist),
                    "feature_dim": features.shape[1],
                })
                trajectory_idx += 1

            except Exception as e:
                print(f"  Warning: Failed for gamma={gamma}, Omega={Omega}: {e}")

    with open(os.path.join(config["data"]["output_dir"], "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"  Generated {len(metadata_list)} trajectories\n")

    # Write modified config to temp file
    temp_config_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # Train
    print("[2/4] Training LSTM on decay dynamics...")
    from src.training.train import train

    model, history = train(
        config_path=temp_config_path,
        data_dir=config["data"]["output_dir"],
        model_dir=os.path.join(output_dir, "models"),
        log_dir=os.path.join(output_dir, "logs"),
    )

    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Evaluate
    print("\n[3/4] Evaluating shelf-life prediction...")
    from src.training.evaluate import (
        load_trained_model,
        reconstruct_hamiltonian,
        evaluate_model,
    )
    from src.data.dataset import QuantumBatteryDataset
    import torch

    device = torch.device("cpu")
    checkpoint_path = os.path.join(output_dir, "models", "best_model.pt")
    model_pt, checkpoint = load_trained_model(checkpoint_path, device)

    d = checkpoint.get("d", 2)
    feature_mean = checkpoint.get("feature_mean", None)
    feature_std = checkpoint.get("feature_std", None)
    H_S_tensor = reconstruct_hamiltonian(omega_0=omega_0, num_tls=num_tls, d=d)

    dataset = QuantumBatteryDataset(
        data_dir=config["data"]["output_dir"],
        window_size=config["data"].get("window_size", 20),
        train=False,
        train_split=0.1,  # Use most data for evaluation
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    summary = evaluate_model(
        model_pt, dataset, H_S_tensor, device, n_trajectories=len(metadata_list),
        feature_mean=feature_mean, feature_std=feature_std,
    )

    # Compute shelf-life: time until stored energy < 0.9 * E_max
    # Note: energies in trajectory files are compute_energy (bare energy).
    # For shelf-life we use stored_energy stored in ergotropies field (hack).
    # Or compute from bare energy: stored = bare + ω₀/2 = bare + 0.5
    print("\n[4/4] Computing storage lifetimes...")
    shelf_lives = []

    for r in summary["results"]:
        # Convert bare energy to stored energy
        bare_energies = np.array(r["true_energies"])
        stored_true = bare_energies + omega_0 / 2.0  # stored = bare + |E_ground| = bare + 0.5
        e_max = np.max(stored_true)
        threshold = 0.9 * e_max
        times = np.array(r["times"])

        # Find first time stored energy drops below threshold
        window_size = len(times) - len(r["predicted_energies"])
        below = np.where(stored_true[window_size:] < threshold)[0]
        shelf_time_true = times[window_size + below[0]] if len(below) > 0 else times[-1]

        pred_bare = np.array(r["predicted_energies"])
        pred_stored = pred_bare + omega_0 / 2.0
        below_pred = np.where(pred_stored < threshold)[0]
        shelf_time_pred = times[window_size + below_pred[0]] if len(below_pred) > 0 else times[-1]

        gamma = r["spectral"]["gamma"]
        memory_time = 1.0 / gamma if gamma > 0 else float("inf")

        shelf_lives.append({
            "gamma": gamma,
            "memory_time": memory_time,
            "shelf_life_true": float(shelf_time_true),
            "shelf_life_pred": float(shelf_time_pred),
        })
        print(f"  gamma={gamma:.3f} (tau_m={memory_time:.2f}): "
              f"shelf_true={shelf_time_true:.2f}, shelf_pred={shelf_time_pred:.2f}")

    with open(os.path.join(output_dir, "shelf_lives.json"), "w", encoding="utf-8") as f:
        json.dump(shelf_lives, f, indent=2)

    # Plot shelf-life vs gamma
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gammas = [s["gamma"] for s in shelf_lives]
    shelf_true = [s["shelf_life_true"] for s in shelf_lives]
    shelf_pred = [s["shelf_life_pred"] for s in shelf_lives]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, shelf_true, "bo-", label="True Shelf-life", markersize=8)
    ax.plot(gammas, shelf_pred, "rs--", label="Predicted Shelf-life", markersize=8)
    ax.set_xlabel("Spectral Width γ")
    ax.set_ylabel("Shelf-life (time to <90% E_max)")
    ax.set_title("Storage Lifetime vs Spectral Width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shelf_life_vs_gamma.png"), dpi=150)
    plt.close()

    print(f"\n[OK] Experiment B complete. Results in: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment B: Storage Lifetime Prediction"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="./experiments/output/experiment_b")
    parser.add_argument("--num-tls", type=int, default=None, help="Number of TLS (default: from config)")
    args = parser.parse_args()
    if args.num_tls is not None:
        pass
    run_experiment_b(args.config, args.output, num_tls=args.num_tls)
