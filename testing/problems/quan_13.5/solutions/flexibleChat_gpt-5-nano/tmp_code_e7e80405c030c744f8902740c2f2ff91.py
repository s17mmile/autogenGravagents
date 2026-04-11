from typing import Tuple

CM1_PER_EV: float = 8065.544  # conversion factor: cm-1 per eV


def compute_De(D0_eV: float, omega_e_cm1: float, omega_e_xe_cm1: float, cm1_per_eV: float = CM1_PER_EV) -> Tuple[float, float]:
    """Compute De in cm-1 and eV using Morse-based zero-point energy approximation.

    D0_eV: dissociation energy from v = 0 in eV
    omega_e_cm1: vibrational constant in cm-1
    omega_e_xe_cm1: anharmonic constant in cm-1
    cm1_per_eV: conversion factor from eV to cm-1 (default 8065.544)

    Returns: (De_cm1, De_eV)
    """
    if D0_eV is None or omega_e_cm1 is None or omega_e_xe_cm1 is None:
        raise ValueError("Inputs must not be None")
    if D0_eV < 0 or omega_e_cm1 < 0 or omega_e_xe_cm1 < 0:
        raise ValueError("Inputs must be non-negative")

    # D0 in cm-1
    D0_cm1 = D0_eV * cm1_per_eV
    # Zero-point energy for v = 0 in Morse approximation
    E0_cm1 = omega_e_cm1 / 2.0 - omega_e_xe_cm1 / 4.0
    # De in cm-1
    De_cm1 = D0_cm1 + E0_cm1
    # Convert back to eV
    De_eV = De_cm1 / cm1_per_eV
    return De_cm1, De_eV


if __name__ == "__main__":
    D0_eV = 2.4287
    omega_e_cm1 = 1405.65
    omega_e_xe_cm1 = 23.20

    De_cm1, De_eV = compute_De(D0_eV, omega_e_cm1, omega_e_xe_cm1)
    print(f"De (cm-1) = {De_cm1:.4f}")
    print(f"De (eV) = {De_eV:.6f}")