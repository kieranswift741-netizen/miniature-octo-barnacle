"""
Quantum battery Hamiltonian: N-TLS Dicke model.

Supports:
  - N = 1: single two-level system
  - N > 1: Dicke model with collective coupling
"""

import numpy as np
import qutip as qt


def build_dicke_hamiltonian(
    num_tls: int = 1,
    omega_0: float = 1.0,
    drive_amplitude: float = 1.0,
    t: float = 0.0,
) -> qt.Qobj:
    """
    Build the time-dependent Dicke model Hamiltonian.

    H(t) = H_S + V(t)
    H_S = Σ_i (ħω₀/2) σ_z^{(i)}
    V(t) = F(t) * Σ_i σ_x^{(i)}

    Args:
        num_tls: Number of two-level systems.
        omega_0: Bare transition frequency (ħω₀).
        drive_amplitude: Driving field amplitude F.
        t: Time for time-dependent drive (placeholder for future pulse shaping).

    Returns:
        qutip.Qobj: Total Hamiltonian operator.
    """
    dims = [2] * num_tls

    # System Hamiltonian: H_S = Σ (ω₀/2) σ_z^{(i)}
    H_S = qt.Qobj(np.zeros((2 ** num_tls, 2 ** num_tls)), dims=[dims, dims])
    for i in range(num_tls):
        op_list = [qt.qeye(2)] * num_tls
        op_list[i] = qt.sigmaz()
        H_S += (omega_0 / 2) * qt.tensor(op_list)

    # Driving term: V(t) = F * Σ σ_x^{(i)}
    V = qt.Qobj(np.zeros((2 ** num_tls, 2 ** num_tls)), dims=[dims, dims])
    for i in range(num_tls):
        op_list = [qt.qeye(2)] * num_tls
        op_list[i] = qt.sigmax()
        V += drive_amplitude * qt.tensor(op_list)

    return H_S + V


def build_system_observables(
    num_tls: int = 1,
    omega_0: float = 1.0,
) -> dict:
    """
    Build key observables for the N-TLS system.

    Args:
        num_tls: Number of two-level systems.
        omega_0: Bare transition frequency.

    Returns:
        dict with 'H_S' (bare system Hamiltonian H_S = Σ (ω₀/2) σ_z^(i)).
    """
    dims = [2] * num_tls

    H_S = qt.Qobj(np.zeros((2 ** num_tls, 2 ** num_tls)), dims=[dims, dims])
    for i in range(num_tls):
        op_list = [qt.qeye(2)] * num_tls
        op_list[i] = (omega_0 / 2.0) * qt.sigmaz()
        H_S += qt.tensor(op_list)

    return {"H_S": H_S}


def coupling_operators(
    num_tls: int = 1,
    mode: str = "collective",
) -> list:
    """
    Build system-bath coupling operators.

    For collective mode: single operator A = Σ σ_x^{(i)} (Dicke model).
    For independent mode: one operator per TLS A_i = σ_x^{(i)}.

    Args:
        num_tls: Number of TLS.
        mode: "collective" or "independent".

    Returns:
        list of qutip.Qobj coupling operators.
    """
    dims = [2] * num_tls

    if mode == "collective":
        A = qt.Qobj(np.zeros((2 ** num_tls, 2 ** num_tls)), dims=[dims, dims])
        for i in range(num_tls):
            op_list = [qt.qeye(2)] * num_tls
            op_list[i] = qt.sigmax()
            A += qt.tensor(op_list)
        return [A]
    else:
        ops = []
        for i in range(num_tls):
            op_list = [qt.qeye(2)] * num_tls
            op_list[i] = qt.sigmax()
            ops.append(qt.tensor(op_list))
        return ops


if __name__ == "__main__":
    H = build_dicke_hamiltonian(num_tls=1, omega_0=1.0, drive_amplitude=1.0, t=0)
    print(f"N=1 Hamiltonian dims: {H.dims}")
    print(f"H_S eigenvalues: {H.eigenenergies()}")

    ops = coupling_operators(num_tls=1, mode="collective")
    print(f"Coupling operators: {len(ops)}, shape {ops[0].shape}")
