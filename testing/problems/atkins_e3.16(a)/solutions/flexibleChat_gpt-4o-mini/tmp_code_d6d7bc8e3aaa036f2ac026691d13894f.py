import math

# This code calculates the change in Gibbs free energy (ΔG) for the expansion of nitrogen gas (N2).

# Constants
n = 3.0e-3  # moles of N2 (3.0 mmol)
R = 8.314  # J/(mol·K)
T = 300  # Temperature in K
V_i = 36e-6  # Initial volume in m³ (36 cm³)
V_f = 60e-6  # Final volume in m³ (60 cm³)

# Error handling for volumes
if V_i <= 0 or V_f <= 0:
    raise ValueError('Volumes must be positive.')

# Step 1: Calculate ΔS
delta_S = n * R * math.log(V_f / V_i)

# Step 2: Calculate ΔH (for isothermal process, ΔH = 0)
delta_H = 0  # J

# Step 3: Calculate ΔG
delta_G = delta_H - T * delta_S

# Output the results
print(f'Change in Entropy (Delta S): {delta_S:.4f} J/K')
print(f'Change in Gibbs Free Energy (Delta G): {delta_G:.4f} J')