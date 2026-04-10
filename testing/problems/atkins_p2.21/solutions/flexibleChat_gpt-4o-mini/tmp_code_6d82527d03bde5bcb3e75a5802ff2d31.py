import numpy as np
from scipy.optimize import fsolve

# Constants
R = 0.0821  # dm^3·atm/(K·mol)
T = 300  # K
P_initial = 500  # atm
P_final = 1  # atm

a = 1.352  # dm^6·atm·mol^-2
b = 0.0387  # dm^3·mol^-1
C_p_m = (7/2) * R  # molar heat capacity at constant pressure

# Function to calculate molar volume using the van der Waals equation
# This function will be used with fsolve to find the root
def van_der_waals_eq(V_m, P, T, a, b):
    return (P + (a / V_m**2)) * (V_m - b) - R * T

# Function to calculate molar volume at a given pressure
def molar_volume(P, T, a, b):
    # Initial guess for molar volume
    V_m_initial_guess = b + (R * T) / P  # Ideal gas approximation
    # Solve the van der Waals equation
    V_m_solution, = fsolve(van_der_waals_eq, V_m_initial_guess, args=(P, T, a, b))
    return V_m_solution  # Return the computed molar volume

# Calculate molar volumes at initial and final pressures
V_m_initial = molar_volume(P_initial, T, a, b)
V_m_final = molar_volume(P_final, T, a, b)

# Calculate mu
mu = ((2 * a / (R * T)) - b) / C_p_m

# Calculate Delta H_m
Delta_P = P_final - P_initial
Delta_H_m = mu * Delta_P

# Output the result
print(f'Delta H_m: {Delta_H_m:.4f} atm·dm^3/mol (equivalent to Joules)')