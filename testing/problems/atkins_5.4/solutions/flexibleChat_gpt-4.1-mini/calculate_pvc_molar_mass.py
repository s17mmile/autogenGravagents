# filename: calculate_pvc_molar_mass.py
import numpy as np
from scipy.stats import linregress

"""
Calculate the molar mass of poly(vinyl chloride) (PVC) from osmotic pressure data.

The osmotic pressure Pi is related to concentration c by Pi = cRT/M for dilute solutions.
Given Pi = rho * g * h, where h is the height of solution column,
this script calculates Pi for each concentration, performs linear regression of Pi/c vs c,
and extrapolates to zero concentration to find M.

Units:
- c in g/dm^3 (equivalent to g/L)
- h in cm (converted to m)
- rho in g/cm^3 (converted to kg/m^3)
- Pi in Pascals (N/m^2)
- R in J/(mol K)
"""

# Given data
c = np.array([1.00, 2.00, 4.00, 7.00, 9.00])  # concentration in g/dm^3
h_cm = np.array([0.28, 0.71, 2.01, 5.10, 8.00])  # height in cm

# Validate input lengths
assert len(c) == len(h_cm), "Concentration and height arrays must be the same length."

# Constants
rho_g_per_cm3 = 0.980  # density in g/cm^3
rho_kg_per_m3 = rho_g_per_cm3 * 1000  # convert to kg/m^3

g = 9.81  # acceleration due to gravity in m/s^2
T = 298  # temperature in K
R = 8.314  # gas constant in J/(mol K)

# Convert height from cm to m
h_m = h_cm / 100

# Calculate osmotic pressure Pi = rho * g * h (in Pascals)
Pi = rho_kg_per_m3 * g * h_m  # Pa

# Calculate Pi/c for each concentration
# Units of Pi/c are Pa * dm^3 / g
Pi_over_c = Pi / c

# Perform linear regression of Pi/c vs c
# According to theory, Pi/c = RT/M + B*c (B is second virial coefficient)
slope, intercept, r_value, p_value, std_err = linregress(c, Pi_over_c)

# Check intercept to avoid division by zero
if abs(intercept) < 1e-12:
    raise ValueError("Intercept too close to zero, cannot calculate molar mass reliably.")

# Calculate molar mass M from intercept: intercept = RT/M => M = RT/intercept
M = R * T / intercept  # g/mol

# Output results
print(f"Linear regression results:")
print(f"Slope (second virial coefficient term): {slope:.5e} Pa*dm^3/g^2")
print(f"Intercept (RT/M): {intercept:.5e} Pa*dm^3/g")
print(f"Calculated molar mass M of PVC: {M:.2f} g/mol")
print(f"Correlation coefficient R^2: {r_value**2:.4f}")

# Save results to a file
with open("pvc_molar_mass_results.txt", "w") as f:
    f.write("Concentration (g/dm^3), Height (cm), Osmotic Pressure (Pa), Pi/c (Pa*dm^3/g)\n")
    for ci, hi, pi_val, pic in zip(c, h_cm, Pi, Pi_over_c):
        f.write(f"{ci}, {hi}, {pi_val:.5f}, {pic:.5f}\n")
    f.write(f"\nLinear regression results:\n")
    f.write(f"Slope: {slope:.5e} Pa*dm^3/g^2\n")
    f.write(f"Intercept: {intercept:.5e} Pa*dm^3/g\n")
    f.write(f"Molar mass M: {M:.2f} g/mol\n")
    f.write(f"R squared: {r_value**2:.4f}\n")
