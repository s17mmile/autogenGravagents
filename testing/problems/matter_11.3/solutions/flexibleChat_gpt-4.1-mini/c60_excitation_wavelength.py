# filename: c60_excitation_wavelength.py
import numpy as np
from scipy.constants import h, m_e, c

# Constants
L = 0.7e-9  # box side length in meters (0.7 nm)
num_electrons = 60  # number of pi electrons

# Generate quantum numbers (n_x, n_y, n_z) for states
max_n = 10  # max quantum number to generate states
states = []
for nx in range(1, max_n+1):
    for ny in range(1, max_n+1):
        for nz in range(1, max_n+1):
            energy_factor = nx**2 + ny**2 + nz**2
            states.append((energy_factor, (nx, ny, nz)))

# Sort states by energy factor
states.sort(key=lambda x: x[0])

# Fill states with electrons (2 per state due to spin)
electrons_placed = 0
occupied_states = []
for energy_factor, quantum_nums in states:
    if electrons_placed + 2 <= num_electrons:
        occupied_states.append((energy_factor, quantum_nums))
        electrons_placed += 2
    else:
        break

# HOMO energy factor is the last occupied state's energy factor
homo_energy_factor = occupied_states[-1][0]

# Find the next higher energy state (LUMO)
lumo_energy_factor = None
for energy_factor, quantum_nums in states:
    if energy_factor > homo_energy_factor:
        lumo_energy_factor = energy_factor
        break

# Calculate energies in Joules
prefactor = h**2 / (8 * m_e * L**2)
E_HOMO = prefactor * homo_energy_factor
E_LUMO = prefactor * lumo_energy_factor

# Excitation energy
delta_E = E_LUMO - E_HOMO

# Convert excitation energy to wavelength (meters)
lambda_m = h * c / delta_E

# Convert to nanometers
lambda_nm = lambda_m * 1e9

# Output the predicted wavelength
print(f"Predicted excitation wavelength for C60: {lambda_nm:.1f} nm")
