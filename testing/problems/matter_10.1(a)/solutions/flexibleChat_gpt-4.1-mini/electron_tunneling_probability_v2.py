# filename: electron_tunneling_probability_v2.py
import numpy as np
from scipy.constants import electron_mass, hbar, e

def transmission_probability(V0_eV, E_eV, L_pm):
    """
    Calculate the transmission probability of an electron tunneling through a rectangular barrier.

    Parameters:
    V0_eV : float
        Barrier height in electronvolts (eV).
    E_eV : float
        Electron energy in electronvolts (eV).
    L_pm : float
        Barrier width in picometers (pm).

    Returns:
    float
        Transmission probability (between 0 and 1).
    """
    # Convert units
    V0 = V0_eV * e  # Barrier height in joules
    E = E_eV * e    # Electron energy in joules
    L = L_pm * 1e-12  # Barrier width in meters

    # Check if electron energy is greater than or equal to barrier height
    if E >= V0:
        # For E >= V0, transmission probability approaches 1 (classically allowed)
        return 1.0

    # Calculate decay constant kappa inside the barrier
    # kappa = sqrt(2*m*(V0 - E)) / hbar
    kappa = np.sqrt(2 * electron_mass * (V0 - E)) / hbar

    # Calculate transmission probability
    T = np.exp(-2 * kappa * L)
    return T

# Given parameters
V0_eV = 2.0  # Barrier height in eV
E_eV = 1.5   # Electron energy in eV
L_pm = 100   # Barrier width in picometers

# Calculate transmission probability
T = transmission_probability(V0_eV, E_eV, L_pm)

# Output the result
print(f"Transmission probability T = {T:.5e}")
