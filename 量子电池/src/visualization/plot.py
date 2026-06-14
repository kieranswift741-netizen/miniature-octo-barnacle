"""
Visualization utilities for quantum battery LSTM results.

Generates:
  - Energy evolution curves (predicted vs true)
  - Ergotropy comparison
  - Efficiency vs spectral width γ
  - Training loss history
  - Generalization error heatmaps
  - Forget gate weight analysis
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")

# ---- Chinese font support ----
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _setup_chinese_font():
    """Detect and set a Chinese-capable font for matplotlib."""
    candidate_fonts = ["Songti SC", "Heiti SC", "STSong", "Arial Unicode MS",
                       "PingFang SC", "Heiti TC", "STHeiti", "SimHei", "Microsoft YaHei"]
    for font_name in candidate_fonts:
        for f in fm.fontManager.ttflist:
            if f.name == font_name:
                plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return font_name
    # Fallback: no Chinese font found, warn
    print("Warning: No Chinese font found. Chinese characters may render as boxes.")
    return None


_CN_FONT = _setup_chinese_font()


# ---- Style settings ----
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def plot_energy_evolution(
    times: np.ndarray,
    true_energies: np.ndarray,
    predicted_energies: np.ndarray,
    params: dict,
    save_path: Optional[str] = None,
):
    """Plot predicted vs true energy evolution for a single trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    window_size = len(times) - len(predicted_energies)

    # Full evolution
    ax1.plot(times, true_energies, "b-", label="True E(t)", linewidth=2)
    pred_times = times[window_size:]
    ax1.plot(pred_times, predicted_energies, "r--", label="Predicted E(t)", linewidth=2)
    ax1.axvline(x=times[window_size], color="gray", linestyle=":", alpha=0.5,
                label=f"Prediction start (t={times[window_size]:.1f})")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy E(t)")
    ax1.set_title(f"Energy Evolution (Ω={params.get('coupling_strength', '?')}, "
                  f"γ={params.get('spectral_width', '?')})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error
    error = np.abs(predicted_energies - true_energies[window_size:])
    ax2.plot(pred_times, error, "purple", linewidth=1.5)
    ax2.fill_between(pred_times, 0, error, alpha=0.3, color="purple")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Prediction Error |E_pred - E_true|")
    ax2.grid(True, alpha=0.3)

    max_err = np.max(error)
    ax2.text(0.95, 0.95, f"Max Error: {max_err:.4f}\nMSE: {np.mean(error**2):.6f}",
             transform=ax2.transAxes, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_ergotropy_comparison(
    times: np.ndarray,
    true_ergotropies: np.ndarray,
    predicted_ergotropies: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot ergotropy comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    window_size = len(times) - len(predicted_ergotropies)
    pred_times = times[window_size:]

    ax.plot(times, true_ergotropies, "g-", label="True Ergotropy", linewidth=2)
    ax.plot(pred_times, predicted_ergotropies, "orange", linestyle="--",
            label="Predicted Ergotropy", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Ergotropy")
    ax.set_title("Extractable Work (Ergotropy) Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: List[dict],
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", h.get("train_total", 0)) for h in history]
    val_loss = [h.get("val_loss", h.get("val_total", 0)) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Main loss
    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Physics constraint losses (if available)
    if "train_trace" in history[0]:
        trace_train = [h.get("train_trace", 0) for h in history]
        herm_train = [h.get("train_herm", 0) for h in history]
        ax2.plot(epochs, trace_train, label="Trace Constraint", linewidth=1)
        ax2.plot(epochs, herm_train, label="Hermiticity", linewidth=1)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Constraint Loss")
        ax2.set_title("Physics Constraint Losses")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No physics constraint data",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Physics Constraints")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_efficiency_vs_gamma(
    evaluation_results: dict,
    save_path: Optional[str] = None,
):
    """
    Plot charging power P vs γ, with one subplot per fixed Ω value.
    
    Each subplot shows how spectral width γ affects power P for a fixed Ω.
    Only Ω values with ≥5 valid γ points are shown.
    """
    from collections import defaultdict
    import math

    # Group results by Ω (coupling strength)
    omega_groups = defaultdict(lambda: {"gammas": [], "eff_true": [], "eff_pred": [], "mse": []})

    for r in evaluation_results.get("results", []):
        spectral = r.get("spectral", {})
        params = r.get("params", {})
        gamma = spectral.get("gamma", params.get("spectral_width", None))
        omega = spectral.get("Omega", params.get("coupling_strength", None))
        if gamma is None or omega is None:
            continue

        # Use peak efficiency (power) if available, otherwise mean
        eff_true = r.get("peak_efficiency_true", np.mean(r["efficiency_true"]))
        eff_pred = r.get("peak_efficiency_pred", np.mean(r["efficiency_pred"]))
        err = r["energy_mse"]

        g = omega_groups[omega]
        g["gammas"].append(gamma)
        g["eff_true"].append(eff_true)
        g["eff_pred"].append(eff_pred)
        g["mse"].append(err)

    if not omega_groups:
        print("No omega data found in evaluation results")
        return

    # Filter: only keep Ω values with ≥5 valid γ points
    valid_omega = {w: g for w, g in omega_groups.items() if len(g["gammas"]) >= 5}
    
    if not valid_omega:
        print(f"  Warning: No Ω values have ≥5 valid γ points. Skipping plot.")
        print(f"  Available: {[(w, len(g['gammas'])) for w, g in omega_groups.items()]}")
        return

    # Sort Ω values
    sorted_omegas = sorted(valid_omega.keys())
    n_omegas = len(sorted_omegas)

    # Layout: 3 columns, enough rows
    n_cols = 3
    n_rows = math.ceil(n_omegas / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle("Charging Power P vs Spectral Width γ  (one subplot per Ω)", fontsize=15, y=1.01)

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "P", "*", "X", "R"]

    for idx, omega in enumerate(sorted_omegas):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        g = valid_omega[omega]
        # Sort by γ
        sort_idx = np.argsort(g["gammas"])
        gammas_arr = np.array(g["gammas"])[sort_idx]
        eff_true_arr = np.array(g["eff_true"])[sort_idx]
        eff_pred_arr = np.array(g["eff_pred"])[sort_idx]
        mse_arr = np.array(g["mse"])[sort_idx]

        mk = markers[idx % len(markers)]
        ax.plot(gammas_arr, eff_true_arr, f"b{mk}-", label="True P", markersize=6, linewidth=1.5)
        ax.plot(gammas_arr, eff_pred_arr, f"r{mk}--", label="Pred P", markersize=6, linewidth=1.5, alpha=0.7)

        n_pts = len(gammas_arr)
        ax.set_title(f"Ω={omega:.3f}  (n={n_pts})", fontsize=11)
        ax.set_xlabel("γ")
        ax.set_ylabel("P")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.invert_xaxis()  # Strong non-Markov (small γ) on right

    # Hide unused subplots
    for idx in range(n_omegas, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_efficiency_vs_gamma_summary(
    evaluation_results: dict,
    save_path: Optional[str] = None,
):
    """
    Summary plot: efficiency vs γ grouped by fixed Ω (two-panel).
    Left: η(γ) curves per Ω.  Right: prediction MSE(γ) per Ω.
    """
    from collections import defaultdict

    omega_groups = defaultdict(lambda: {"gammas": [], "eff_true": [], "eff_pred": [], "mse": []})

    for r in evaluation_results.get("results", []):
        spectral = r.get("spectral", {})
        params = r.get("params", {})
        gamma = spectral.get("gamma", params.get("spectral_width", None))
        omega = spectral.get("Omega", params.get("coupling_strength", None))
        if gamma is None or omega is None:
            continue

        # Use peak efficiency if available, otherwise mean of trajectory efficiency
        eff_true = r.get("peak_efficiency_true", np.mean(r["efficiency_true"]))
        eff_pred = r.get("peak_efficiency_pred", np.mean(r["efficiency_pred"]))
        err = r["energy_mse"]

        g = omega_groups[omega]
        g["gammas"].append(gamma)
        g["eff_true"].append(eff_true)
        g["eff_pred"].append(eff_pred)
        g["mse"].append(err)

    if not omega_groups:
        print("No gamma data found in evaluation results")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(omega_groups)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for ci, (omega, g) in enumerate(sorted(omega_groups.items())):
        idx = np.argsort(g["gammas"])
        gammas_arr = np.array(g["gammas"])[idx]
        eff_true_arr = np.array(g["eff_true"])[idx]
        eff_pred_arr = np.array(g["eff_pred"])[idx]
        mse_arr = np.array(g["mse"])[idx]

        mk = markers[ci % len(markers)]
        label = f"Ω={omega:.2f}"

        ax1.plot(gammas_arr, eff_true_arr, color=colors[ci], marker=mk,
                 linestyle="-", label=f"{label} (true)", markersize=6)
        ax1.plot(gammas_arr, eff_pred_arr, color=colors[ci], marker=mk,
                 linestyle="--", label=f"{label} (pred)", markersize=6, alpha=0.7)

        ax2.plot(gammas_arr, mse_arr, color=colors[ci], marker=mk,
                 linestyle="-", label=label, markersize=6)

    ax1.set_xlabel("Spectral Width γ")
    ax1.set_ylabel("Charging Power P")
    ax1.set_title("Charging Power vs Spectral Width (grouped by Ω)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2.set_xlabel("Spectral Width γ")
    ax2.set_ylabel("Prediction MSE")
    ax2.set_title("Prediction Error vs Spectral Width (grouped by Ω)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_generalization_heatmap(
    train_params: List[dict],
    test_errors: List[dict],
    param_x: str = "coupling_strength",
    param_y: str = "spectral_width",
    save_path: Optional[str] = None,
):
    """Plot generalization error heatmap across parameter space."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = sorted(set(d.get(param_x, 0) for d in test_errors))
    y_vals = sorted(set(d.get(param_y, 0) for d in test_errors))

    error_grid = np.zeros((len(y_vals), len(x_vals)))
    for d in test_errors:
        xi = x_vals.index(d[param_x])
        yi = y_vals.index(d[param_y])
        error_grid[yi, xi] = d.get("energy_mse", 0)

    im = ax.imshow(error_grid, origin="lower", aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{x:.2f}" for x in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{y:.2f}" for y in y_vals])
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title("Generalization Error Across Parameter Space")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Energy MSE")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_forget_gate_analysis(
    forget_weights: np.ndarray,
    feature_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """Analyze LSTM forget gate weights to infer environmental memory time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(forget_weights.flatten(), bins=30, color="teal", alpha=0.7, edgecolor="black")
    ax1.axvline(x=np.mean(forget_weights), color="red", linestyle="--",
                label=f"Mean: {np.mean(forget_weights):.3f}")
    ax1.axvline(x=np.median(forget_weights), color="orange", linestyle="--",
                label=f"Median: {np.median(forget_weights):.3f}")
    ax1.set_xlabel("Forget Gate Weight Value")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Forget Gate Weight Distribution")
    ax1.legend()

    mean_per_feature = np.mean(np.abs(forget_weights), axis=0)
    if feature_labels is None:
        feature_labels = [f"f{i}" for i in range(len(mean_per_feature))]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(mean_per_feature)))
    ax2.barh(feature_labels[:len(mean_per_feature)], mean_per_feature, color=colors)
    ax2.set_xlabel("Mean |Weight|")
    ax2.set_title("Forget Gate Sensitivity per Input Feature")
    ax2.axvline(x=np.mean(mean_per_feature), color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_summary_dashboard(
    eval_summary: dict,
    output_dir: str,
):
    """Generate all plots from evaluation results and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for i, r in enumerate(eval_summary["results"]):
        plot_energy_evolution(
            times=np.array(r["times"]),
            true_energies=np.array(r["true_energies"]),
            predicted_energies=np.array(r["predicted_energies"]),
            params=r["params"],
            save_path=os.path.join(output_dir, f"energy_traj_{i:02d}.png"),
        )

        plot_ergotropy_comparison(
            times=np.array(r["times"]),
            true_ergotropies=np.array(r["true_ergotropies"]),
            predicted_ergotropies=np.array(r["predicted_ergotropies"]),
            save_path=os.path.join(output_dir, f"ergotropy_traj_{i:02d}.png"),
        )

    plot_efficiency_vs_gamma(
        eval_summary,
        save_path=os.path.join(output_dir, "efficiency_vs_gamma.png"),
    )

    print(f"Saved {len(eval_summary['results']) * 2 + 1} plots to {output_dir}")


if __name__ == "__main__":
    times = np.linspace(0, 20, 400)
    true_e = 1 - np.exp(-0.3 * times) * np.cos(0.5 * times)
    pred_e = true_e[-200:] + np.random.normal(0, 0.02, 200)

    test_dir = "/tmp/plots_test"
    os.makedirs(test_dir, exist_ok=True)

    plot_energy_evolution(
        times, true_e, pred_e,
        params={"coupling_strength": 0.1, "spectral_width": 0.5},
        save_path=os.path.join(test_dir, "test_energy.png"),
    )
    print(f"Test plot saved to {test_dir}")
    if _CN_FONT:
        print(f"Chinese font in use: {_CN_FONT}")
