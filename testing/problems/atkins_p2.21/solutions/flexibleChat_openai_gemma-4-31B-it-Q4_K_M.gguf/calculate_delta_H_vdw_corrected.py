# filename: calculate_delta_H_vdw_corrected.py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

# Constants
R = 0.082057  # L·atm·mol⁻¹·K⁻¹
a = 1.352     # L²·atm·mol⁻² (dm⁶·atm·mol⁻²)
b = 0.0387    # L·mol⁻¹ (dm³·mol⁻¹)
T = 300       # K
P1 = 500      # atm (initial)
P2 = 1.00     # atm (final)

# Define function to solve for Vm given P using root_scalar
def solve_Vm(P):
    def f(V):
        return (P + a / V**2) * (V - b) - R * T
    # Initial guess: ideal gas volume
    V_guess = R * T / P
    # Use bounded root finder to ensure physical root
    try:
        sol = root_scalar(f, bracket=[V_guess * 0.5, V_guess * 2], method='bisect')
        if not sol.converged:
            raise ValueError("Root not converged")
        return sol.root
    except Exception as e:
        raise ValueError(f"Failed to solve for Vm at P = {P} atm: {e}")

# Define integrand: (Vm - b)
def integrand(P):
    Vm = solve_Vm(P)
    return Vm - b

# Compute integral from P2 to P1 (1 to 500 atm), then negate
delta_H_m, error = quad(integrand, P2, P1)
delta_H_m = -delta_H_m  # because dP is negative in original direction

# Output result
print(f"ΔH_m = {delta_H_m:.4f} L·atm·mol⁻¹")

# Convert to J·mol⁻¹ (1 L·atm = 101.325 J)
delta_H_m_J = delta_H_m * 101.325
print(f"ΔH_m = {delta_H_m_J:.4f} J·mol⁻¹")