import numpy as np

# Constants
lambda_charge = 3.6e-9  # Linear charge density in C/m
epsilon_0 = 8.85e-12    # Permittivity of free space in C^2/(N m^2)
radius_shell = 0.015     # Radius of the cylindrical shell in m

# Step 1: Calculate the electric field due to the charged wire
E_wire = lambda_charge / (2 * np.pi * epsilon_0 * radius_shell)

# Step 2: Calculate the surface charge density sigma
sigma = epsilon_0 * E_wire

# Output the result
print(f'Surface charge density (sigma): {sigma:.2e} C/m^2')