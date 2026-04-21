# filename: calculate_entropy_change.py

import math

def calculate_entropy_change(P1_bar, P2_bar, T1, T2, beta, rho, M, Cp):
    """
    Calculate the entropy change (Delta S) for one mole of liquid water
    undergoing a transformation from (P1, T1) to (P2, T2).

    Parameters:
    P1_bar (float): Initial pressure in bar
    P2_bar (float): Final pressure in bar
    T1 (float): Initial temperature in Kelvin
    T2 (float): Final temperature in Kelvin
    beta (float): Thermal expansion coefficient in 1/K
    rho (float): Density in kg/m^3
    M (float): Molar mass in kg/mol
    Cp (float): Heat capacity at constant pressure in J/(mol K)

    Returns:
    float: Entropy change in J/K
    """
    # Basic input validation
    if P1_bar <= 0 or P2_bar <= 0:
        raise ValueError("Pressures must be positive.")
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperatures must be positive.")
    if rho <= 0 or M <= 0 or Cp <= 0:
        raise ValueError("Density, molar mass, and heat capacity must be positive.")

    # Convert pressures from bar to Pascals
    P1 = P1_bar * 1e5  # Pa
    P2 = P2_bar * 1e5  # Pa

    # Calculate molar volume (m^3/mol)
    V = M / rho

    # Calculate entropy change using the formula
    Delta_S = Cp * math.log(T2 / T1) - V * beta * (P2 - P1)

    return Delta_S


# Given data
P1_bar = 1.00  # bar
P2_bar = 590.0  # bar
T1 = 350.0  # K
T2 = 750.0  # K
beta = 2.07e-4  # 1/K
rho = 997.0  # kg/m^3
M = 0.018015  # kg/mol
Cp = 75.3  # J/(mol K)

# Calculate and print the entropy change
Delta_S = calculate_entropy_change(P1_bar, P2_bar, T1, T2, beta, rho, M, Cp)
print(f"Change in entropy (Delta S) = {Delta_S:.4f} J/K")
