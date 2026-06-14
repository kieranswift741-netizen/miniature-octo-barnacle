"""
Physics-informed loss functions for quantum battery LSTM.

Combines data-driven MSE with physical constraints:
  - Trace conservation: Tr(ρ) must equal 1
  - Hermiticity: ρ must equal ρ†
  - Positivity: eigenvalues of ρ must be non-negative

Total Loss = L_MSE + λ₁ L_trace + λ₂ L_herm + λ₃ L_pos
"""

import torch
import torch.nn as nn
from typing import Optional


def reconstruct_density_matrix(
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct complex density matrices from flattened real+imag output.

    Input shape:  (batch, 2 * d²)  where first  half = real parts,
                                         second half = imag parts.
    Output shape: (batch, d, d) as complex tensors.

    Args:
        y: Real-valued tensor of shape (batch, 2 * d²).

    Returns:
        Complex tensor of shape (batch, d, d).
    """
    batch_size = y.shape[0]
    half_dim = y.shape[1] // 2
    d = int(half_dim ** 0.5)
    assert d * d == half_dim, f"Dimension mismatch: half_dim={half_dim}, d² expected, got d={d}"

    real_part = y[:, :half_dim].reshape(batch_size, d, d)
    imag_part = y[:, half_dim:].reshape(batch_size, d, d)

    return torch.complex(real_part, imag_part)


def trace_constraint(
    rho: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize deviation from Tr(ρ) = 1.

    L_trace = |Tr(ρ) - 1|²

    Args:
        rho: Complex density matrix (batch, d, d).

    Returns:
        Scalar loss averaged over batch.
    """
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)  # (batch,)
    return torch.mean(torch.abs(trace - 1.0) ** 2)


def hermiticity_constraint(
    rho: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize deviation from hermiticity.

    L_herm = ||ρ - ρ†||² / d²

    Args:
        rho: Complex density matrix (batch, d, d).

    Returns:
        Scalar loss averaged over batch.
    """
    diff = rho - rho.conj().transpose(-2, -1)
    norm_sq = torch.sum(diff.real ** 2 + diff.imag ** 2, dim=(-2, -1))
    d = rho.shape[-1]
    return torch.mean(norm_sq) / (d * d)


def positivity_constraint(
    rho: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize negative eigenvalues.

    L_pos = Σ_i max(-λ_i, 0)

    Args:
        rho: Complex density matrix (batch, d, d).

    Returns:
        Scalar loss averaged over batch.
    """
    eigenvalues = torch.linalg.eigvalsh(rho)
    negative_parts = torch.clamp(-eigenvalues, min=0.0)
    return torch.mean(negative_parts.sum(dim=-1))


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for quantum density matrix prediction.

    Loss = L_MSE + λ_trace * L_trace + λ_herm * L_herm + λ_pos * L_pos
    """

    def __init__(
        self,
        lambda_trace: float = 1.0,
        lambda_herm: float = 1.0,
        lambda_pos: float = 0.1,
        use_physics_constraints: bool = True,
        d: int = 2,
    ):
        """
        Args:
            lambda_trace: Weight for trace constraint.
            lambda_herm: Weight for hermiticity constraint.
            lambda_pos: Weight for positivity constraint.
            use_physics_constraints: If False, only MSE is used.
            d: Hilbert space dimension.
        """
        super().__init__()
        self.lambda_trace = lambda_trace
        self.lambda_herm = lambda_herm
        self.lambda_pos = lambda_pos
        self.use_physics_constraints = use_physics_constraints
        self.d = d
        self.mse = nn.MSELoss()
        self._last_losses = {}

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            y_pred: Predicted flattened density matrix (batch, 2*d²).
            y_true: True flattened density matrix (batch, 2*d²).

        Returns:
            Total scalar loss.
        """
        # Data-driven loss
        l_mse = self.mse(y_pred, y_true)
        self._last_losses["mse"] = l_mse.item()

        total_loss = l_mse

        if self.use_physics_constraints:
            # Reconstruct density matrices
            rho_pred = reconstruct_density_matrix(y_pred)

            # Physics constraints
            l_trace = trace_constraint(rho_pred)
            l_herm = hermiticity_constraint(rho_pred)
            l_pos = positivity_constraint(rho_pred)

            self._last_losses["trace"] = l_trace.item()
            self._last_losses["herm"] = l_herm.item()
            self._last_losses["pos"] = l_pos.item()

            total_loss = (
                l_mse
                + self.lambda_trace * l_trace
                + self.lambda_herm * l_herm
                + self.lambda_pos * l_pos
            )

        self._last_losses["total"] = total_loss.item()
        return total_loss

    def get_component_losses(self) -> dict:
        """Return the loss components from the last forward pass."""
        return self._last_losses.copy()


def energy_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    H_S_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE on energy derived from predicted density matrix.

    E = Tr(ρ H_S), then MSE on energy values.

    Args:
        y_pred: Predicted flattened ρ (batch, 2*d²).
        y_true: True flattened ρ (batch, 2*d²).
        H_S_matrix: Hamiltonian matrix (d, d) as complex tensor.

    Returns:
        Energy MSE loss.
    """
    rho_pred = reconstruct_density_matrix(y_pred)
    rho_true = reconstruct_density_matrix(y_true)

    # Compute energies: E = Tr(ρ H_S)
    energy_pred = torch.real(
        torch.einsum("bij,jk,bki->b", rho_pred, H_S_matrix, torch.eye(H_S_matrix.shape[0]).unsqueeze(0).expand(rho_pred.shape[0], -1, -1))
    )
    # Simpler: E = sum_i ρ_ii * H_ii (if H is diagonal) or full trace
    energy_pred = torch.real(torch.diagonal(
        torch.matmul(rho_pred, H_S_matrix.unsqueeze(0)), dim1=-2, dim2=-1
    ).sum(dim=-1))

    energy_true = torch.real(torch.diagonal(
        torch.matmul(rho_true, H_S_matrix.unsqueeze(0)), dim1=-2, dim2=-1
    ).sum(dim=-1))

    return nn.functional.mse_loss(energy_pred, energy_true)


if __name__ == "__main__":
    # Test with dummy data (N=1, d=2)
    d = 2
    output_dim = 2 * d * d  # 8

    # Create a valid density matrix
    rho_real = torch.tensor([[0.7, 0.1], [0.1, 0.3]], dtype=torch.float32)
    rho_imag = torch.tensor([[0.0, 0.05], [-0.05, 0.0]], dtype=torch.float32)

    y_true = torch.cat([
        rho_real.flatten(),
        rho_imag.flatten(),
    ]).unsqueeze(0)  # (1, 8)

    # Perfect prediction
    y_pred = y_true.clone()

    criterion = PhysicsInformedLoss(d=d)
    loss = criterion(y_pred, y_true)
    print(f"Perfect prediction loss: {loss.item():.6f}")
    print(f"Component losses: {criterion.get_component_losses()}")

    # Bad prediction
    y_bad = torch.randn(1, output_dim)
    loss_bad = criterion(y_bad, y_true)
    print(f"\nRandom prediction loss: {loss_bad.item():.6f}")
    print(f"Component losses: {criterion.get_component_losses()}")
