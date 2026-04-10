import numpy as np
from scipy.optimize import fsolve

# Constants
P = 327.6  # Pressure in atm
T = 776.4  # Temperature in K
R = 0.0821  # Ideal gas constant in dm³·atm/(K·mol)
a = 5.464  # van der Waals constant a in dm^6·atm·mol^-2
b = 0.03049  # van der Waals constant b in dm^3·mol^-1

# Function to calculate the van der Waals equation
# V_m is the molar volume in dm³/mol

def van_der_waals(V_m):
    """
    Calculate the van der Waals equation for a given molar volume.
    :param V_m: Molar volume in dm³/mol
    :return: Value of the van der Waals equation
    """
    n = 1  # number of moles
    return (P + (a * n**2) / V_m**2) * (V_m - n * b) - n * R * T

# Initial guess for molar volume using ideal gas law
V_m_initial = (R * T) / P

# Solve for molar volume using fsolve
try:
    V_m_solution = fsolve(van_der_waals, V_m_initial)
    molar_volume = V_m_solution[0]
    if molar_volume <= 0:
        raise ValueError('Calculated molar volume is non-positive.')
    print(f'Molar Volume of water vapor: {molar_volume:.3f} dm³/mol')
except Exception as e:
    print(f'Error in calculation: {e}')