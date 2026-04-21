# filename: calculate_molar_volume.py
import numpy as np
from scipy.optimize import root_scalar

# Given data
P = 327.6  # atm
T = 776.4  # K
rho = 133.2  # kg/m^3
Tc = 647.4  # K
pc = 218.3  # atm

a = 5.464  # dm^6 atm mol^-2 (L^2 atm mol^-2)
b = 0.03049  # dm^3 mol^-1 (L mol^-1)
M = 0.01802  # kg/mol

# Gas constant in L atm / mol K
R = 0.08206

# Step 1: Calculate molar density (mol/m^3)
molar_density = rho / M  # mol/m^3

# Step 2: Calculate molar volume from density (m^3/mol)
Vm_m3 = 1 / molar_density  # m^3/mol

# Convert molar volume to liters per mole (1 m^3 = 1000 L)
Vm_density = Vm_m3 * 1000  # L/mol

# Step 3: Define Van der Waals equation function to solve for Vm
# Vm in L/mol

def vdw_eq(Vm):
    return (P + a / Vm**2) * (Vm - b) - R * T

# Dynamically set bracket for root finding
lower_bound = b + 1e-6
upper_bound = max(1.5 * Vm_density, lower_bound + 1.0)  # ensure upper bound > lower bound

# Solve Van der Waals equation numerically
sol = root_scalar(vdw_eq, bracket=[lower_bound, upper_bound], method='brentq')

if not sol.converged:
    raise RuntimeError("Failed to solve Van der Waals equation for molar volume.")

Vm_vdw = sol.root

# Output results
print(f"Molar volume from density: {Vm_density:.4f} L/mol")
print(f"Molar volume from Van der Waals equation: {Vm_vdw:.4f} L/mol")
