# filename: calculate_gibbs_energy_difference.py
import math

def gibbs_energy_difference(T, phi):
    """Calculate the difference in molar Gibbs energy between a real gas and a perfect gas.

    Parameters:
    T (float): Temperature in Kelvin
    phi (float): Fugacity coefficient (must be positive)

    Returns:
    float: Difference in molar Gibbs energy in Joules per mole
    """
    if phi <= 0:
        raise ValueError("Fugacity coefficient must be positive.")
    R = 8.314  # J/(mol K), universal gas constant
    delta_G_m = R * T * math.log(phi)
    return delta_G_m

# Given values
T = 200.0  # Temperature in K
phi = 0.72  # Fugacity coefficient

delta_G_m = gibbs_energy_difference(T, phi)

# Output the result
print(f"Difference in molar Gibbs energy (J/mol): {delta_G_m:.2f}")
