"""
Lorentzian (underdamped Brownian oscillator) spectral density
for structured reservoirs with non-Markovian effects.

J(ω) = (1 / 2π) * γ * Ω² / ((ω - ω_c)² + γ²)

The bath correlation function at T=0:
C(t) = (Ω² / 2) * exp(-γ|t|) * exp(-i ω_c t)
     = (Ω² / 2) * exp(-γ|t|) * [cos(ω_c t) - i sin(ω_c t)]
"""

import numpy as np
from typing import Tuple


class LorentzianSpectralDensity:
    """Lorentzian spectral density for structured (non-Markovian) baths."""

    def __init__(
        self,
        coupling_strength: float = 0.1,  # Ω
        spectral_width: float = 0.5,     # γ (smaller = stronger memory)
        center_frequency: float = 1.0,   # ω_c
        temperature: float = 0.0,        # T (0 = vacuum)
    ):
        """
        Args:
            coupling_strength: System-bath coupling strength Ω.
            spectral_width: Spectral width γ. Smaller γ → stronger non-Markovianity.
            center_frequency: Central frequency ω_c of the reservoir mode.
            temperature: Bath temperature (currently only T=0 supported).
        """
        self.Omega = coupling_strength
        self.gamma = spectral_width
        self.omega_c = center_frequency
        self.temperature = temperature

    def spectral_density(self, omega: np.ndarray) -> np.ndarray:
        """
        Evaluate J(ω) at given frequencies.

        J(ω) = (1 / 2π) * γ * Ω² / ((ω - ω_c)² + γ²)
        """
        return (1.0 / (2.0 * np.pi)) * self.gamma * self.Omega ** 2 / (
            (omega - self.omega_c) ** 2 + self.gamma ** 2
        )

    def correlation_function(self, t: np.ndarray) -> np.ndarray:
        """
        Bath correlation function C(t) at T=0.

        C(t) = (Ω² / 2) * exp(-γ|t|) * exp(-i ω_c t)
        """
        return (self.Omega ** 2 / 2.0) * np.exp(
            -self.gamma * np.abs(t) - 1j * self.omega_c * t
        )

    def memory_time(self) -> float:
        """
        Estimate the bath memory time τ_m = 1 / γ.
        Larger τ_m → stronger non-Markovian effects.
        """
        return 1.0 / self.gamma if self.gamma > 0 else float("inf")

    def non_markovianity_measure(self) -> float:
        """
        Heuristic measure of non-Markovianity.
        Returns Ω / γ: larger ratio → stronger memory effects.
        """
        return self.Omega / self.gamma if self.gamma > 0 else float("inf")

    def to_dict(self) -> dict:
        """Export spectral density parameters for metadata."""
        return {
            "type": "Lorentzian",
            "Omega": self.Omega,
            "gamma": self.gamma,
            "omega_c": self.omega_c,
            "temperature": self.temperature,
            "memory_time": self.memory_time(),
            "non_markovianity": self.non_markovianity_measure(),
        }

    def __repr__(self) -> str:
        return (
            f"LorentzianSpectralDensity(Ω={self.Omega}, γ={self.gamma}, "
            f"ω_c={self.omega_c}, T={self.temperature})"
        )


def lorentzian_exponents(
    coupling_strength: float,
    spectral_width: float,
    center_frequency: float,
) -> Tuple[complex, complex, complex]:
    """
    Return exponential decomposition coefficients for the Lorentzian
    bath correlation function, suitable for QuTiP HEOM.

    C(t) = Σ c_k * exp(-ν_k * t)

    For Lorentzian at T=0:
    c₁ = Ω²/2,  ν₁ = γ + i ω_c
    c₂ = Ω²/2,  ν₂ = γ - i ω_c

    Returns:
        Tuple of (c, ν_real, ν_imag) for QuTiP's exponential bath.
    """
    ck = coupling_strength ** 2 / 2.0
    nu_real = spectral_width
    nu_imag = center_frequency
    return complex(ck, 0), complex(nu_real, 0), complex(nu_imag, 0)


if __name__ == "__main__":
    sd = LorentzianSpectralDensity(Omega=0.1, gamma=0.2, omega_c=1.0)

    omega = np.linspace(0, 3, 100)
    J = sd.spectral_density(omega)
    print(f"J(ω_c) = {sd.spectral_density(np.array([1.0]))[0]:.4f}")
    print(f"Memory time τ_m = {sd.memory_time():.3f}")
    print(f"Non-Markovianity Ω/γ = {sd.non_markovianity_measure():.3f}")

    t = np.array([0, 1, 2])
    C = sd.correlation_function(t)
    print(f"C(t=0) = {C[0]:.4f}")
