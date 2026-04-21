# filename: calculate_temperature_two_level_system.py
import numpy as np
from scipy.constants import h, c, k, N_A
from scipy.optimize import root_scalar

# Given constants
energy_wavenumber_cm = 1000  # in cm^-1
internal_energy_target = 3.00e3  # 3.00 kJ in Joules

# Convert wavenumber to energy in Joules
# E = h * c * wavenumber
# c in m/s, wavenumber in cm^-1, so convert cm^-1 to m^-1 by multiplying by 100
energy_J = h * c * energy_wavenumber_cm * 100  # Joules

# Check that energy difference is positive
assert energy_J > 0, "Energy difference must be positive"

# Define the function for internal energy difference from target
# U = N_A * (DeltaE / (1 + exp(beta * DeltaE)))
# beta = 1/(k*T)
# We want U - internal_energy_target = 0

def internal_energy_difference(T):
    if T <= 0:
        raise ValueError("Temperature must be positive")
    beta = 1 / (k * T)
    avg_energy_per_particle = energy_J / (1 + np.exp(beta * energy_J))
    U = N_A * avg_energy_per_particle
    return U - internal_energy_target

# Use root finding to solve for T
# Temperature bracket chosen from 1 K to 5000 K as physically reasonable range
solution = root_scalar(internal_energy_difference, bracket=[1, 5000], method='brentq')

if solution.converged:
    temperature = solution.root
    print(f'Temperature at which internal energy equals 3.00 kJ: {temperature:.2f} K')
else:
    temperature = None
    print('Failed to find temperature solution.')

# Save the result to a file
with open('temperature_result.txt', 'w') as f:
    if temperature is not None:
        f.write(f'Temperature at which internal energy equals 3.00 kJ: {temperature:.2f} K\n')
    else:
        f.write('Failed to find temperature solution.\n')
