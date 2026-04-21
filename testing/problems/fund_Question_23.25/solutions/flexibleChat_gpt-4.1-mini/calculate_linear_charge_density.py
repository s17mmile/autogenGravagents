# filename: calculate_linear_charge_density.py
import math

def calculate_linear_charge_density(E, r):
    """
    Calculate the linear charge density (lambda) of an infinite line of charge.
    Parameters:
        E (float): Electric field magnitude in N/C (must be positive).
        r (float): Distance from the line charge in meters (must be positive).
    Returns:
        float: Linear charge density in Coulombs per meter.
    Raises:
        ValueError: If E or r is not positive.
    """
    if E <= 0:
        raise ValueError("Electric field magnitude E must be positive.")
    if r <= 0:
        raise ValueError("Distance r must be positive.")

    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(N m^2)
    lambda_linear = E * 2 * math.pi * epsilon_0 * r
    return lambda_linear

# Given values
E = 4.5e4  # Electric field in N/C
r = 2.0    # Distance in meters

lambda_linear = calculate_linear_charge_density(E, r)
print(f"Linear charge density (lambda): {lambda_linear:.3e} C/m")
