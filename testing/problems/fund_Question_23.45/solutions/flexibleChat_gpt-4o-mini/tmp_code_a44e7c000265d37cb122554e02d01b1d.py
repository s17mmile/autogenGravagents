import numpy as np

# Constants
epsilon_0 = 8.85e-12  # Permittivity of free space in C^2/(N m^2)

def calculate_electric_field(Q, r):
    """
    Calculate the electric field due to a charged spherical shell.
    Parameters:
    Q : float : Charge in coulombs
    r : float : Distance from the center in meters
    Returns:
    float : Electric field in kN/C
    """
    if r <= 0:
        raise ValueError('Radius must be greater than zero.')
    E = (1 / (4 * np.pi * epsilon_0)) * (Q / r**2)  # Electric field in N/C
    return E / 1000  # Convert to kN/C

# Charge on inner shell
Q1 = 4.00e-8  # Charge in C
r = 0.12  # Distance from center in meters (12.0 cm)

# Calculate the electric field
E_kN_C = calculate_electric_field(Q1, r)

# Print the result
print(f'Electric field at r = 12.0 cm: {E_kN_C:.2f} kN/C')