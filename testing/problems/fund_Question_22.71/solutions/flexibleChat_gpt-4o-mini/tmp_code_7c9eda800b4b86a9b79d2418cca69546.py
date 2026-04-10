import numpy as np

# Constants
Q = 20e-9  # Charge in Coulombs
L = 4.0    # Length of the rod in meters
R = 2.0    # Radius of the arc in meters
k = 8.99e9 # Coulomb's constant in N m^2/C^2

# Input validation
if Q < 0 or L <= 0 or R <= 0:
    raise ValueError('Charge must be positive and length/radius must be positive.')

# Linear charge density
linear_charge_density = Q / L

# Electric field contribution from a small element
# The arc subtends an angle of theta = L / R
theta_total = L / R

# Total electric field calculation
# The formula derived from integration is used directly here
E_total = (k * linear_charge_density / R) * (theta_total)

# Print the result with units
print(f'Magnitude of the electric field at the center of curvature: {E_total:.2f} N/C')