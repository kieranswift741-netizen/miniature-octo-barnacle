"""
Non-Markovian open quantum system dynamics using QuTiP HEOM solver.

Wraps the Hierarchical Equations of Motion (HEOM) solver to generate
time-series density matrix data for training the LSTM model.
"""

import numpy as np
from typing import Optional
import qutip as qt

from .spectral_density import LorentzianSpectralDensity


def build_bosonic_bath(
    spectral_density: LorentzianSpectralDensity,
    Q,
) -> "qt.heom.BosonicBath":
    """
    Build a QuTiP BosonicBath from Lorentzian spectral density.

    Uses the UnderDampedBath model to represent the Lorentzian
    bath in exponential decomposition form.

    Args:
        spectral_density: Lorentzian spectral density parameters.
        Q: System coupling operator (e.g., Σ σ_x for collective coupling).

    Returns:
        BosonicBath object for HEOM solver.
    """
    gamma = spectral_density.gamma
    # lam = Omega^2 / 2: matches Lorentzian C(t=0) = Omega^2/2 in UnderDampedBath at T=0
    lam = spectral_density.Omega ** 2 / 2.0
    w0 = spectral_density.omega_c
    T = spectral_density.temperature
    Nk = 2

    bath = qt.heom.UnderDampedBath(
        Q,
        gamma,
        lam,
        w0,
        T,
        Nk,
        tag="structured_bath",
    )

    return bath


def run_heom_simulation(
    H_S,
    coupling_ops: list,
    spectral_density: LorentzianSpectralDensity,
    rho_0=None,
    tlist: Optional[np.ndarray] = None,
    max_depth: int = 5,
    options=None,
) -> dict:
    """
    Run HEOM simulation for the quantum battery in a structured bath.

    Args:
        H_S: System Hamiltonian (time-independent part).
        coupling_ops: List of coupling operators A_i.
        spectral_density: Lorentzian spectral density parameters.
        rho_0: Initial density matrix (default: all ground |0...0⟩).
        tlist: Time points for output.
        max_depth: HEOM truncation level.
        options: QuTiP solver options.

    Returns:
        dict with 'times', 'states', 'params'.
    """
    if rho_0 is None:
        num_tls = len(H_S.dims[0])
        dims = [2] * num_tls
        rho_0 = qt.ket2dm(qt.basis(2 ** num_tls, 0))
        rho_0.dims = [dims, dims]

    if tlist is None:
        tlist = np.arange(0, 20.0, 0.05)

    if options is None:
        options = None

    # Build baths from spectral density
    baths = []
    for Q in coupling_ops:
        bath = build_bosonic_bath(spectral_density, Q)
        baths.append(bath)

    # Setup HEOM solver
    solver = qt.heom.HEOMSolver(H_S, baths, max_depth=max_depth, options=options)

    # Run evolution
    result = solver.run(rho_0, tlist)

    # Extract density matrices
    states = []
    for s in result.states:
        if s.isket:
            states.append(qt.ket2dm(s))
        else:
            states.append(s)

    return {
        "times": tlist,
        "states": states,
        "params": spectral_density.to_dict(),
    }


def density_matrix_to_features(rho) -> np.ndarray:
    """
    Convert a density matrix to a real-valued feature vector.

    Extracts the real and imaginary parts of all matrix elements,
    flattened into a 1D array.

    Args:
        rho: Density matrix as QuTiP Qobj.

    Returns:
        1D numpy array of shape (2 * d²,) where d = dim(Hilbert space).
    """
    full = rho.full()
    real_part = full.real.flatten()
    imag_part = full.imag.flatten()
    return np.concatenate([real_part, imag_part])


def compute_energy(rho, H_S) -> float:
    """
    Compute bare system energy E(t) = Tr[ρ(t) H_S].

    For a TLS with H_S = (ω₀/2) σ_z, this ranges from -ω₀/2 (ground) to +ω₀/2 (excited).
    Use stored_energy() for battery-specific energy measured from ground state.

    Args:
        rho: Density matrix at time t.
        H_S: Bare system Hamiltonian.

    Returns:
        Bare energy value.
    """
    return np.real(qt.expect(H_S, rho))


def stored_energy(rho, H_S) -> float:
    """
    Compute battery stored energy = Tr[ρ(t) H_S] - E_ground.

    Shifts the bare energy so that the ground state corresponds to E=0.
    This is the physically meaningful measure of how much energy the
    battery has stored relative to its fully discharged state.

    Args:
        rho: Density matrix at time t.
        H_S: Bare system Hamiltonian.

    Returns:
        Stored energy (non-negative).
    """
    eigenvalues = H_S.eigenenergies()
    E_ground = np.min(np.real(eigenvalues))
    return np.real(qt.expect(H_S, rho)) - E_ground


def compute_ergotropy(rho, H_S) -> float:
    """
    Compute ergotropy: maximum extractable work from a quantum state.

    E(ρ) = Tr(ρ H_S) - Tr(σ_ρ H_S)

    where σ_ρ is the passive state.

    Args:
        rho: Density matrix.
        H_S: System Hamiltonian.

    Returns:
        Ergotropy value (non-negative).
    """
    eigenvalues, eigenvectors = H_S.eigenstates()
    idx = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = [eigenvectors[i] for i in idx]

    populations = np.array(
        [np.real(qt.expect(qt.ket2dm(ev), rho)) for ev in eigenvectors_sorted]
    )

    rho_eigenvalues = np.sort(np.real(rho.eigenenergies()))[::-1]
    passive_energy = np.sum(rho_eigenvalues * eigenvalues_sorted)
    internal_energy = np.real(qt.expect(H_S, rho))

    return max(0.0, internal_energy - passive_energy)


if __name__ == "__main__":
    from .hamiltonian import build_dicke_hamiltonian, coupling_operators

    H_S = build_dicke_hamiltonian(num_tls=1, omega_0=1.0, drive_amplitude=0.0, t=0)
    ops = coupling_operators(num_tls=1, mode="collective")
    sd = LorentzianSpectralDensity(coupling_strength=0.1, spectral_width=0.2, center_frequency=1.0)

    print(f"Hamiltonian dims: {H_S.dims}")
    print(f"Coupling ops: {len(ops)}")
    print(f"Spectral density: {sd}")

    result = run_heom_simulation(H_S, ops, sd, max_depth=3)
    print(f"Simulated {len(result['times'])} time steps")
    print(f"ρ(t=0) shape: {result['states'][0].shape}")
    print(f"Energy at t=10: {compute_energy(result['states'][200], H_S):.6f}")
    print(f"Ergotropy at t=10: {compute_ergotropy(result['states'][200], H_S):.6f}")
