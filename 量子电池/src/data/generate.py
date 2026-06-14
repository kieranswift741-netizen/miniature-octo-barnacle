"""
Data generation pipeline: parameter sweep over structured reservoir configurations
and batch HEOM simulations to produce training data for the LSTM model.
"""

import os
import sys
import json
import argparse
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    compute_ergotropy,
)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file, with sensible defaults."""
    config = {
        "physics": {
            "num_tls": 1,
            "omega_0": 1.0,
            "drive_amplitude": 1.0,
        },
        "heom": {
            "max_depth": 5,
            "time_step": 0.05,
            "total_time": 20.0,
        },
        "data": {
            "output_dir": "./data/generated",
            "param_grid": {
                "coupling_strength": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                "spectral_width": [0.1, 0.2, 0.5, 1.0],
            },
            "num_trajectories": 500,
            "window_size": 20,
        },
    }

    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        # Deep merge
        for section in yaml_config:
            if section in config and isinstance(config[section], dict):
                config[section].update(yaml_config[section])
            else:
                config[section] = yaml_config[section]

    return config


def generate_single_trajectory(
    H_S,
    coupling_ops,
    spectral_density: LorentzianSpectralDensity,
    config: dict,
    rho_0=None,
) -> dict:
    """
    Run a single HEOM trajectory and extract features + metrics.

    Returns:
        dict with 'times', 'features', 'energies', 'ergotropies', 'params'.
    """
    tlist = np.arange(
        0,
        config["heom"]["total_time"],
        config["heom"]["time_step"],
    )

    result = run_heom_simulation(
        H_S=H_S,
        coupling_ops=coupling_ops,
        spectral_density=spectral_density,
        rho_0=rho_0,
        tlist=tlist,
        max_depth=config["heom"]["max_depth"],
    )

    states = result["states"]
    omega_0 = config["physics"].get("omega_0", 1.0)
    H_S_bare = build_system_observables(config["physics"]["num_tls"], omega_0=omega_0)["H_S"]

    features = np.array([density_matrix_to_features(rho) for rho in states])
    energies = np.array([compute_energy(rho, H_S_bare) for rho in states])
    ergotropies = np.array([compute_ergotropy(rho, H_S_bare) for rho in states])

    # Validate physicality with stricter checks
    num_tls = config["physics"]["num_tls"]
    omega_0 = config["physics"].get("omega_0", 1.0)
    
    # Check 1: Density matrix elements must be bounded (theoretical max = 1.0)
    max_abs_feat = np.max(np.abs(features))
    feat_bound = 1.1  # Allow 10% numerical tolerance
    
    # Check 2: Energy must be within physical bounds
    energy_bound = num_tls * omega_0 * 1.1  # 10% tolerance
    max_abs_energy = np.max(np.abs(energies))
    
    # Check 3: Trace conservation (Tr(ρ) = 1)
    max_trace_deviation = max(
        abs(np.trace(rho.full()) - 1.0) for rho in states
    )
    
    # Check 4: Positivity (eigenvalues >= 0)
    min_eigenvalue = min(
        np.min(np.linalg.eigvals(rho.full()).real) for rho in states
    )
    
    # Report violations
    violations = []
    if max_abs_feat > feat_bound:
        violations.append(f"max|ρ_ij|={max_abs_feat:.3f} (bound={feat_bound:.2f})")
    if max_abs_energy > energy_bound:
        violations.append(f"max|E|={max_abs_energy:.3f} (bound={energy_bound:.2f})")
    if max_trace_deviation > 1e-3:
        violations.append(f"trace deviation={max_trace_deviation:.2e}")
    if min_eigenvalue < -1e-3:
        violations.append(f"min eigenvalue={min_eigenvalue:.2e}")
    
    if violations:
        raise RuntimeError(
            f"Unphysical trajectory: {'; '.join(violations)}. "
            f"HEOM may not be converged — increase max_depth "
            f"(current: {config['heom']['max_depth']})."
        )

    return {
        "times": tlist,
        "features": features,
        "energies": energies,
        "ergotropies": ergotropies,
        "params": spectral_density.to_dict(),
        "num_tls": config["physics"]["num_tls"],
        "omega_0": config["physics"]["omega_0"],
    }


def generate_dataset(
    config: dict,
    output_dir: str,
    seeds: Optional[list] = None,
) -> list:
    """
    Generate the full training dataset by sweeping over parameter grid.

    Args:
        config: Configuration dictionary.
        output_dir: Directory to save generated data.
        seeds: Optional list of random seeds for initial states.

    Returns:
        list of metadata dicts for all generated trajectories.
    """
    os.makedirs(output_dir, exist_ok=True)

    physics_cfg = config["physics"]
    param_grid = config["data"]["param_grid"]

    # Build system Hamiltonian (time-independent)
    H_S = build_dicke_hamiltonian(
        num_tls=physics_cfg["num_tls"],
        omega_0=physics_cfg["omega_0"],
        drive_amplitude=physics_cfg["drive_amplitude"],
        t=0.0,
    )

    coupling_mode = "collective" if physics_cfg["num_tls"] > 1 else "collective"
    ops = coupling_operators(num_tls=physics_cfg["num_tls"], mode=coupling_mode)

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    n_total = len(combinations)
    print(f"Generating {n_total} trajectories over parameter grid:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")

    metadata_list = []
    trajectory_idx = 0

    for combo in tqdm(combinations, desc="Generating trajectories"):
        params = dict(zip(param_names, combo))

        sd = LorentzianSpectralDensity(
            coupling_strength=params.get("coupling_strength", 0.1),
            spectral_width=params.get("spectral_width", 0.5),
            center_frequency=physics_cfg.get("center_frequency", 1.0),
        )

        try:
            traj = generate_single_trajectory(H_S, ops, sd, config)

            # Save trajectory data
            filepath = os.path.join(output_dir, f"traj_{trajectory_idx:04d}.npz")
            np.savez_compressed(
                filepath,
                times=traj["times"],
                features=traj["features"],
                energies=traj["energies"],
                ergotropies=traj["ergotropies"],
            )

            meta = {
                "file": filepath,
                "index": trajectory_idx,
                "params": params,
                "spectral": traj["params"],
                "num_time_steps": len(traj["times"]),
                "feature_dim": traj["features"].shape[1],
            }
            metadata_list.append(meta)
            trajectory_idx += 1

        except Exception as e:
            print(f"\nWarning: Failed for {params}: {e}")

    # Save metadata index
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\nSaved {trajectory_idx} trajectories to {output_dir}")
    print(f"Metadata index: {meta_path}")

    return metadata_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate quantum battery evolution data via HEOM simulations."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file (default: config.yaml)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (overrides config)."
    )
    parser.add_argument(
        "--num-tls", type=int, default=None,
        help="Override number of TLS."
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Override HEOM truncation depth."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.num_tls is not None:
        config["physics"]["num_tls"] = args.num_tls
    if args.max_depth is not None:
        config["heom"]["max_depth"] = args.max_depth

    output_dir = args.output or config["data"]["output_dir"]

    generate_dataset(config, output_dir)


if __name__ == "__main__":
    main()
