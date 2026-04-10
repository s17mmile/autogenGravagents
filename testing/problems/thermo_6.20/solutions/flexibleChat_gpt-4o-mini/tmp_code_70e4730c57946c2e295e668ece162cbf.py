import numpy as np

# Constants
n_moles = 2.25  # moles of gas
R = 8.314  # J/(mol*K)
T = 325  # K
initial_pressure = 12.0 * 1e5  # initial pressure in Pa (12.0 bar)
final_pressure = 2.5 * 1e5  # final pressure in Pa (2.5 bar)

# Calculate change in entropy (ΔS)
Delta_S = n_moles * R * np.log(initial_pressure / final_pressure)

# Calculate change in Gibbs free energy (ΔG)
Delta_G = -T * Delta_S

# Convert ΔG to kJ
Delta_G_kJ = Delta_G / 1000

# Print the result with appropriate significant figures
print(f'Delta G approximately {Delta_G_kJ:.3f} kJ')