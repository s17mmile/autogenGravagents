# filename: chemical_potential_change.py
import math

def calculate_delta_mu(P1, P2, T_C):
    """
    Calculate the change in chemical potential (delta_mu) of a perfect gas
    when its pressure changes isothermally from P1 to P2 at temperature T_C.

    Parameters:
    P1 (float): Initial pressure in atm (must be > 0)
    P2 (float): Final pressure in atm (must be > 0)
    T_C (float): Temperature in degrees Celsius

    Returns:
    float: Change in chemical potential in Joules per mole (J/mol)
    """
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Pressures must be positive values.")
    R = 8.3145  # J/(mol K)
    T = T_C + 273.15  # Convert Celsius to Kelvin
    delta_mu = R * T * math.log(P2 / P1)
    return delta_mu

# Given values
P1 = 1.8  # atm
P2 = 29.5  # atm
T_C = 40  # degrees Celsius

# Calculate change in chemical potential
delta_mu = calculate_delta_mu(P1, P2, T_C)

# Print the result
print(f"Change in chemical potential (delta_mu) = {delta_mu:.2f} J/mol")
