#!/usr/bin/env python3
"""
Experiment C: Cross-Parameter Extrapolation (Generalization)

Tests the LSTM's ability to predict battery behavior at coupling strengths
NOT seen during training. This demonstrates the model's capacity to learn
the underlying physics rather than memorizing trajectories.

Key question: Can an LSTM trained on Ω ∈ [0.05, 0.4] accurately predict
behavior at Ω = 0.5, 0.6?

Method:
  1. Split coupling strengths into train set and test set
  2. Train LSTM only on the train set
  3. Evaluate prediction accuracy on unseen test coupling strengths
  4. Compare in-distribution vs out-of-distribution error
"""

import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics.hamiltonian import build_dicke_hamiltonian, coupling_operators
from src.physics.spectral_density import LorentzianSpectralDensity
from src.data.generate import load_config
from src.data.dataset import QuantumBatteryDataset, create_dataloaders
from src.visualization.plot import plot_generalization_heatmap, plot_energy_evolution


def run_experiment_c(
    config_path: str = "config.yaml",
    output_dir: str = "./experiments/output/experiment_c",
    num_tls: int = None,
):
    """
    Run Experiment C: Cross-parameter generalization.

    1. Generate full parameter sweep data
    2. Split into train (Ω ∈ [0.05, 0.4]) and test (Ω ∈ [0.5, 0.6])
    3. Train on train set only
    4. Evaluate generalization on unseen Ω values
    """
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)

    # Override number of TLS if specified
    if num_tls is not None:
        config["physics"]["num_tls"] = num_tls

    physics_cfg = config["physics"]

    # Full parameter sweep — denser grid for better generalization test
    all_Omega = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    train_Omega = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    test_Omega = [0.5, 0.6]

    config["data"]["param_grid"] = {
        "coupling_strength": all_Omega,
        "spectral_width": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    }
    config["data"]["output_dir"] = os.path.join(output_dir, "data_all")
    config["heom"]["max_depth"] = 5
    config["training"]["epochs"] = 200
    config["training"]["early_stop_patience"] = 30

    print("=" * 70)
    print("Experiment C: Cross-Parameter Extrapolation (Generalization)")
    print("=" * 70)
    print(f"N_TLS = {physics_cfg['num_tls']}, omega_0 = {physics_cfg['omega_0']}")
    print(f"Train Omega: {train_Omega}")
    print(f"Test  Omega: {test_Omega}")
    print()

    # Step 1: Generate all data
    print("[1/5] Generating full parameter sweep data...")
    from src.data.generate import generate_dataset

    metadata_all = generate_dataset(config, config["data"]["output_dir"])
    print(f"  Generated {len(metadata_all)} total trajectories\n")

    # Step 2: Split into train and test directories by coupling strength
    print("[2/5] Splitting data into train/test by coupling strength...")
    train_dir = os.path.join(output_dir, "data_train")
    test_dir = os.path.join(output_dir, "data_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_meta = []
    test_meta = []
    train_idx = 0
    test_idx = 0

    for meta in metadata_all:
        Omega_val = meta["params"]["coupling_strength"]
        src_file = meta["file"]

        if Omega_val in train_Omega:
            dest = os.path.join(train_dir, f"traj_{train_idx:04d}.npz")
            shutil.copy2(src_file, dest)
            new_meta = dict(meta)
            new_meta["file"] = dest
            new_meta["index"] = train_idx
            train_meta.append(new_meta)
            train_idx += 1
        else:
            dest = os.path.join(test_dir, f"traj_{test_idx:04d}.npz")
            shutil.copy2(src_file, dest)
            new_meta = dict(meta)
            new_meta["file"] = dest
            new_meta["index"] = test_idx
            test_meta.append(new_meta)
            test_idx += 1

    with open(os.path.join(train_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)
    with open(os.path.join(test_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2)

    print(f"  Train: {len(train_meta)} trajectories")
    print(f"  Test:  {len(test_meta)} trajectories\n")

    # Write modified config to temp file
    temp_config_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # Step 3: Train on train set only
    print("[3/5] Training LSTM on train coupling strengths only...")
    from src.training.train import train

    model, history = train(
        config_path=temp_config_path,
        data_dir=train_dir,
        model_dir=os.path.join(output_dir, "models"),
        log_dir=os.path.join(output_dir, "logs"),
    )

    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Step 4: Evaluate on both train and test distributions
    print("\n[4/5] Evaluating generalization...")
    from src.training.evaluate import (
        load_trained_model,
        reconstruct_hamiltonian,
        evaluate_model,
    )
    import torch

    device = torch.device("cpu")
    checkpoint_path = os.path.join(output_dir, "models", "best_model.pt")
    model_pt, checkpoint = load_trained_model(checkpoint_path, device)

    d = checkpoint.get("d", 2)
    omega_0 = physics_cfg.get("omega_0", 1.0)
    feature_mean = checkpoint.get("feature_mean", None)
    feature_std = checkpoint.get("feature_std", None)
    H_S_tensor = reconstruct_hamiltonian(omega_0=omega_0, num_tls=physics_cfg["num_tls"], d=d)

    window_size = config["data"].get("window_size", 20)

    # In-distribution evaluation
    ds_train = QuantumBatteryDataset(
        data_dir=train_dir,
        window_size=window_size,
        train=False,
        train_split=0.01,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    summary_train = evaluate_model(
        model_pt, ds_train, H_S_tensor, device, n_trajectories=5,
        feature_mean=feature_mean, feature_std=feature_std,
    )

    # Out-of-distribution evaluation
    ds_test = QuantumBatteryDataset(
        data_dir=test_dir,
        window_size=window_size,
        train=False,
        train_split=0.01,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    summary_test = evaluate_model(
        model_pt, ds_test, H_S_tensor, device, n_trajectories=len(test_meta),
        feature_mean=feature_mean, feature_std=feature_std,
    )

    # Combine results
    generalization_results = {
        "train_distribution": {"Omega_range": train_Omega},
        "test_distribution": {"Omega_range": test_Omega},
        "in_distribution_mse": summary_train["avg_energy_mse"],
        "out_of_distribution_mse": summary_test["avg_energy_mse"],
        "generalization_ratio": (
            summary_test["avg_energy_mse"] / summary_train["avg_energy_mse"]
            if summary_train["avg_energy_mse"] > 0 else float("inf")
        ),
        "in_dist_results": summary_train["results"],
        "out_dist_results": summary_test["results"],
    }

    with open(os.path.join(output_dir, "generalization.json"), "w", encoding="utf-8") as f:
        json.dump(generalization_results, f, indent=2, default=str)

    # Step 5: Visualize
    print("\n[5/5] Generating generalization plots...")

    # Build test error list for heatmap
    test_errors = []
    for r in summary_test["results"]:
        test_errors.append({
            "coupling_strength": r["params"].get("coupling_strength", 0),
            "spectral_width": r["params"].get("spectral_width", 0),
            "energy_mse": r["energy_mse"],
        })

    plot_generalization_heatmap(
        train_params=[{"coupling_strength": o, "spectral_width": 0.5} for o in train_Omega],
        test_errors=test_errors,
        param_x="coupling_strength",
        param_y="spectral_width",
        save_path=os.path.join(output_dir, "generalization_heatmap.png"),
    )

    # Summary
    print("\n" + "=" * 70)
    print("Generalization Results:")
    print("-" * 70)
    print(f"  In-distribution  (Omega in {train_Omega}): MSE = {summary_train['avg_energy_mse']:.6f}")
    print(f"  Out-of-distribution (Omega in {test_Omega}): MSE = {summary_test['avg_energy_mse']:.6f}")
    print(f"  Generalization Ratio: {generalization_results['generalization_ratio']:.3f}")
    print(f"  (Ratio < 2.0 → good generalization; > 5.0 → poor generalization)")

    for r in summary_test["results"]:
        print(f"\n  Omega={r['params']['coupling_strength']}, gamma={r['params']['spectral_width']}:")
        print(f"    Energy MSE = {r['energy_mse']:.6f}")

    print(f"\n[OK] Experiment C complete. Results in: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment C: Cross-Parameter Extrapolation"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="./experiments/output/experiment_c")
    parser.add_argument("--num-tls", type=int, default=None, help="Number of TLS (default: from config)")
    args = parser.parse_args()
    if args.num_tls is not None:
        pass
    run_experiment_c(args.config, args.output, num_tls=args.num_tls)
