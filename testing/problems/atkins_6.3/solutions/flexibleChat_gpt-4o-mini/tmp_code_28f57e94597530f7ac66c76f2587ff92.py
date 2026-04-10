import numpy as np
from scipy.stats import linregress

# Given data
T = np.array([350, 400, 450, 500])  # Temperature in K
K = np.array([3.98e-4, 1.41e-2, 1.86e-1, 1.48])  # Equilibrium constants

# Error handling: Check if K contains non-positive values
if np.any(K <= 0):
    raise ValueError('Equilibrium constants must be positive.')

# Calculate ln(K)
lnK = np.log(K)

# Calculate 1/T
invT = 1 / T

# Perform linear regression to find slope
slope, intercept, r_value, p_value, std_err = linregress(invT, lnK)

# Calculate standard reaction enthalpy
R = 8.314  # J/(mol*K)
standard_reaction_enthalpy_J = -slope * R
standard_reaction_enthalpy_kJ = standard_reaction_enthalpy_J / 1000  # Convert to kJ/mol

# Output the result
print(f'Standard reaction enthalpy (Delta H) = {standard_reaction_enthalpy_kJ:.2f} kJ/mol')  # Delta H is derived from the slope of ln(K) vs 1/T