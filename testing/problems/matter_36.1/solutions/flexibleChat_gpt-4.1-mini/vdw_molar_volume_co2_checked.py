# filename: vdw_molar_volume_co2_checked.py
import numpy as np
from scipy.optimize import fsolve

# Constants for CO2
R = 0.08206  # L atm / mol K
T = 500      # K
P = 100      # atm

a = 3.59     # L^2 atm / mol^2
b = 0.0427   # L / mol

# Define the van der Waals equation as a function of molar volume Vm
# (P + a / Vm^2) * (Vm - b) = R * T

def vdw_eq(Vm):
    return (P + a / Vm**2) * (Vm - b) - R * T

# Initial guess for Vm (ideal gas approximation)
Vm_initial = R * T / P

# Solve for Vm
Vm_solution, info, ier, mesg = fsolve(vdw_eq, Vm_initial, full_output=True)

if ier == 1 and Vm_solution[0] > 0:
    print(f"Estimated molar volume of CO2 at {T} K and {P} atm: {Vm_solution[0]:.4f} L/mol")
else:
    print("Failed to find a physically meaningful molar volume solution.")
    print("Solver message:", mesg)
