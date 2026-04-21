# filename: energy_uncertainty_calculation.py

import math

# Constants
h = 6.62607015e-34  # Planck constant in J*s
hbar = h / (2 * math.pi)  # Reduced Planck constant (h-bar)

# Function to calculate energy uncertainty given lifetime
# Using the energy-time uncertainty principle:
# Delta E * Delta t >= hbar / 2  =>  Delta E >= hbar / (2 * Delta t)
def calculate_energy_uncertainty(lifetime_seconds):
    return hbar / (2 * lifetime_seconds)

# Given lifetime of excited state
lifetime = 1e-9  # seconds

# Calculate energy uncertainty
delta_E = calculate_energy_uncertainty(lifetime)

print(f"Uncertainty in energy (Delta E): {delta_E:.3e} J")
