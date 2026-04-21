# filename: n2_speed_fraction.py
import numpy as np
from scipy.integrate import quad

# Constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
T = 298.0  # Temperature in K
N_A = 6.02214076e23  # Avogadro's number
molar_mass_N2 = 28.0134e-3  # kg/mol

# Calculate mass of one N2 molecule (kg)
m = molar_mass_N2 / N_A

# Maxwell-Boltzmann speed distribution function
# Returns probability density for speed v (m/s)
def maxwell_boltzmann_pdf(v, m, T):
    factor = 4 * np.pi * (m / (2 * np.pi * k_B * T))**1.5
    return factor * v**2 * np.exp(-m * v**2 / (2 * k_B * T))

# Speed range for fraction calculation
v_min = 200.0  # m/s
v_max = 300.0  # m/s

# Numerically integrate PDF over [v_min, v_max] to find fraction
fraction, error = quad(maxwell_boltzmann_pdf, v_min, v_max, args=(m, T))

# Print integration error for transparency
print(f"Numerical integration error estimate: {error:.3e}")

# Simple test: integral over [0, infinity) should be ~1
total_prob, total_error = quad(maxwell_boltzmann_pdf, 0, np.inf, args=(m, T))
print(f"Integral over full speed range: {total_prob:.6f} (error estimate: {total_error:.3e})")

# Output the fraction result
print(f"Fraction of N2 molecules with speed between {v_min} m/s and {v_max} m/s at {T} K: {fraction:.6f}")

# The variable 'fraction' holds the desired fraction value.