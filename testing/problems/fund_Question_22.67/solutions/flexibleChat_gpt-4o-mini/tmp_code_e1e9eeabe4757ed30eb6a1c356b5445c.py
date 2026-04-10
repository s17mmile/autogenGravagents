import numpy as np
from scipy.integrate import quad

# Constants
rho = 9.0e-9  # Linear charge density in C/m
epsilon_0 = 8.85e-12  # Permittivity of free space in C^2/(N m^2)

# Function to calculate the electric field contribution from a small segment of charge
# at position x' to the point at x = 4.0 m
def electric_field_contribution(x_prime):
    r = 4.0 - x_prime  # Distance from charge segment to point of interest
    dq = rho  # Charge per unit length (C/m)
    dE = (1 / (4 * np.pi * epsilon_0)) * (dq / r**2)  # Electric field contribution
    return dE  # This is positive, but we will consider direction in integration

# Integrate the electric field contributions from 0 to 3.0 m
E_total, _ = quad(lambda x: electric_field_contribution(x), 0, 3.0)

# Since the electric field points away from the charge, we take the negative of the total
E_magnitude = -E_total

# Output the result
print(f'Magnitude of the electric field at x = 4.0 m: {E_magnitude:.2e} N/C directed away from the charged line.')