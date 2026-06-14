#!/usr/bin/env python3
"""
Experiment A: Non-Markovian Boosting (Feedback Gain)

Investigates how the LSTM captures energy backflow from the structured reservoir
and identifies parameters where environmental memory enhances charging rate.

Key question: At what γ (spectral width) does non-Markovianity boost the
battery's effective charging rate beyond the Markovian limit?

Method:
  1. Generate trajectories across a fine γ sweep
  2. Train LSTM on all data
  3. Compare LSTM-predicted energy vs Markovian (Lindblad) baseline
  4. Identify γ regime with maximum "boosting" effect
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics.hamiltonian import build_dicke_hamiltonian, coupling_operators
from src.physics.spectral_density import LorentzianSpectralDensity
from src.data.generate import generate_dataset, load_config
from src.visualization.plot import (
    plot_efficiency_vs_gamma,
    plot_efficiency_vs_gamma_summary,
    plot_energy_evolution,
)


def run_experiment_a(
    config_path: str = "config.yaml",
    output_dir: str = "./experiments/output/experiment_a",
    num_tls: int = None,
):
    """
    Run Experiment A: Non-Markovian Boosting analysis.

    1. Generate data with fine γ sweep (small γ = strong memory)
    2. Train model
    3. Analyze boosting effect
    """
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)

    # Override number of TLS if specified
    if num_tls is not None:
        config["physics"]["num_tls"] = num_tls

    # Dense parameter sweep — model needs many points to learn gamma-dependence
    # Dense parameter grid: ensure ≥5 Ω values each with ≥5 valid γ points
    # With ~40% HEOM success rate, need ~180 total points for ~72 successful
    config["data"]["param_grid"] = {
        "coupling_strength": [
            0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
        ],  # 15 points
        "spectral_width": [
            0.02, 0.03, 0.05, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0
        ],  # 15 points
    }
    config["data"]["output_dir"] = os.path.join(output_dir, "data")
    config["data"]["window_size"] = 100  # Increased from 20 to cover longer memory times
    config["heom"]["total_time"] = 15.0  # First charging peak ~t=3, 15 is sufficient
    config["heom"]["max_depth"] = 10  # Increased from 5 for better convergence
    config["training"]["epochs"] = 200
    config["training"]["early_stop_patience"] = 30

    physics_cfg = config["physics"]

    print("=" * 70)
    print("Experiment A: Non-Markovian Boosting (Feedback Gain)")
    print("=" * 70)
    print(f"N_TLS = {physics_cfg['num_tls']}, omega_0 = {physics_cfg['omega_0']}")
    print(f"gamma sweep: {config['data']['param_grid']['spectral_width']}")
    print(f"Omega values: {config['data']['param_grid']['coupling_strength']}")
    print()

    # Step 1: Generate data
    print("[1/4] Generating HEOM trajectories...")
    metadata = generate_dataset(config, config["data"]["output_dir"])
    print(f"  Generated {len(metadata)} trajectories\n")

    # Write modified config to temp file so train() picks up overrides
    temp_config_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # Step 2: Train model
    print("[2/4] Training LSTM model...")
    from src.training.train import train

    model, history = train(
        config_path=temp_config_path,
        data_dir=config["data"]["output_dir"],
        model_dir=os.path.join(output_dir, "models"),
        log_dir=os.path.join(output_dir, "logs"),
    )

    # Save history
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Step 3: Evaluate
    print("\n[3/4] Evaluating model...")
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
    omega_0 = physics_cfg.get("omega_0", 1.0)
    H_S_tensor = reconstruct_hamiltonian(omega_0=omega_0, num_tls=physics_cfg["num_tls"], d=d)

    # Load normalization stats from checkpoint
    feature_mean = checkpoint.get("feature_mean", None)
    feature_std = checkpoint.get("feature_std", None)

    dataset = QuantumBatteryDataset(
        data_dir=config["data"]["output_dir"],
        window_size=config["data"].get("window_size", 20),
        train=False,
        train_split=0.8,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    summary = evaluate_model(
        model_pt, dataset, H_S_tensor, device, n_trajectories=len(dataset.traj_indices),
        feature_mean=feature_mean, feature_std=feature_std,
    )

    with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Step 4: Visualize boosting effect
    print("\n[4/4] Generating plots...")
    
    # Generate per-gamma subplot figure
    plot_path = os.path.join(output_dir, "efficiency_vs_gamma.png")
    plot_efficiency_vs_gamma(summary, save_path=plot_path)
    if os.path.exists(plot_path):
        print(f"  ✓ Generated: {plot_path}")
    else:
        print(f"  ✗ Failed to generate: {plot_path}")
    
    # Generate summary figure (eta vs gamma grouped by Omega)
    summary_path = os.path.join(output_dir, "efficiency_vs_gamma_summary.png")
    plot_efficiency_vs_gamma_summary(summary, save_path=summary_path)
    if os.path.exists(summary_path):
        print(f"  ✓ Generated: {summary_path}")
    else:
        print(f"  ✗ Failed to generate: {summary_path}")

    # Analyze boosting: does smaller γ (stronger memory) give higher efficiency?
    print("\n" + "=" * 70)
    print("Boosting Analysis:")
    print("-" * 70)
    print(f"{'Omega':>8} {'gamma':>8} {'tau_m':>8} {'P_true':>10} {'P_pred':>10} {'t_peak':>8} {'MSE':>10}")
    print("-" * 80)
    for r in sorted(summary["results"], key=lambda x: (x["spectral"].get("gamma", 0), x["spectral"].get("Omega", 0))):
        gamma = r["spectral"]["gamma"]
        omega = r["spectral"]["Omega"]
        tau_m = r["spectral"].get("memory_time", 1.0 / gamma if gamma > 0 else float("inf"))
        power_true = r.get("peak_efficiency_true", 0.0)
        power_pred = r.get("peak_efficiency_pred", 0.0)
        t_peak = r.get("t_peak_true", 0.0)
        mse = r["energy_mse"]
        print(f"{omega:>8.3f} {gamma:>8.3f} {tau_m:>8.2f} {power_true:>10.4f} {power_pred:>10.4f} {t_peak:>8.2f} {mse:>10.6f}")

    print("\n[OK] Experiment A complete. Results in:", output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment A: Non-Markovian Boosting"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="./experiments/output/experiment_a")
    parser.add_argument("--num-tls", type=int, default=None, help="Number of TLS (default: from config)")
    args = parser.parse_args()
    if args.num_tls is not None:
        # Override will happen after load_config in run_experiment_a
        pass
    run_experiment_a(args.config, args.output, num_tls=args.num_tls)
