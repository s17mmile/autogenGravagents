# filename: calculate_electric_field_cylindrical_shells.py

import math

def calculate_electric_field(lambda_inner, r):
    """Calculate electric field magnitude at distance r from a line charge with linear charge density lambda_inner."""
    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(N*m^2)
    E = lambda_inner / (2 * math.pi * epsilon_0 * r)
    return E

# Given values
lambda_inner = 5.0e-6  # C/m
r = 0.04  # m (4.0 cm)

E = calculate_electric_field(lambda_inner, r)

print(f"Electric field magnitude at r = {r} m: {E:.2e} N/C")
