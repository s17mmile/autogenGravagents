#!/usr/bin/env python3
"""
Generalized Bohr-model transition calculator for muonic atoms.
Calculates the n1 -> n2 transition energy, frequency, and wavelength for a muon bound to a nucleus with charge Z.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TransitionResult:
    mu_over_m_e: float
    E1: float
    E2: float
    delta_E_eV: float
    freq_Hz: float
    lambda_nm: float


def compute_muonic_transition(n1: int,
                              n2: int,
                              Z: int = 1,
                              m_p_over_m_e: float = 1836.15267389,
                              m_mu_over_m_e: float = 206.7682837,
                              e_base_eV: float = 13.605693,
                              h_eVs: float = 4.135667696e-15,
                              hc_eV_nm: float = 1239.841984) -> TransitionResult:
    """Compute the Bohr-model transition for a muon bound to a nucleus.

    Parameters:
      n1, n2: Principal quantum numbers (n1 > 0, n2 > 0, n1 != n2)
      Z: Nuclear charge (default 1 for muonic hydrogen)
      m_p_over_m_e: Proton mass in electron mass units (default CODATA-like)
      m_mu_over_m_e: Muon mass in electron mass units (default CODATA-like)
      e_base_eV: Base energy scale for the Bohr model in eV (default 13.605693 eV)
      h_eVs: Planck constant in eV*s (default 4.135667696e-15 eV*s)
      hc_eV_nm: hc in eV*nm (default 1239.841984 eV*nm)

    Returns:
      TransitionResult with mu_over_m_e, E1, E2, delta_E_eV, freq_Hz, lambda_nm
    """
    if not (isinstance(n1, int) and isinstance(n2, int)):
        raise TypeError("n1 and n2 must be integers")
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive integers")
    if n1 == n2:
        raise ValueError("n1 and n2 must be different for a transition")
    if Z <= 0:
        raise ValueError("Z must be a positive integer")

    # Reduced mass factor (in units of m_e)
    mu_over_m_e = (m_p_over_m_e * m_mu_over_m_e) / (m_p_over_m_e + m_mu_over_m_e)

    # Energies (Bohr-like, Z^2 scaling)
    E1 = - mu_over_m_e * e_base_eV * (Z ** 2) / (n1 ** 2)
    E2 = - mu_over_m_e * e_base_eV * (Z ** 2) / (n2 ** 2)

    delta_E_eV = abs(E1 - E2)

    freq_Hz = delta_E_eV / h_eVs
    lambda_nm = hc_eV_nm / delta_E_eV

    return TransitionResult(mu_over_m_e, E1, E2, delta_E_eV, freq_Hz, lambda_nm)


if __name__ == "__main__":
    # Example: muonic hydrogen (Z = 1), 1S -> 2P
    n1, n2, Z = 1, 2, 1
    result = compute_muonic_transition(n1, n2, Z=Z)

    print(f"mu_over_m_e = {result.mu_over_m_e:.12f}")
    print(f"E1 (eV)       = {result.E1:.6f}")
    print(f"E2 (eV)       = {result.E2:.6f}")
    print(f"DeltaE (eV)    = {result.delta_E_eV:.6f}")
    print(f"Frequency (Hz)   = {result.freq_Hz:.3e}")
    print(f"Wavelength (nm)  = {result.lambda_nm:.6f}")