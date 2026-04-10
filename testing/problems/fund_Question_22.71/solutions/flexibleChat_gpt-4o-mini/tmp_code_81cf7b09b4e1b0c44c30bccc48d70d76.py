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

# The arc subtends an angle of theta_total = L / R
theta_total = L / R

# Numerical integration parameters
num_segments = 1000  # Number of segments for integration
theta_values = np.linspace(-theta_total / 2, theta_total / 2, num_segments)

# Initialize total electric field
E_total = 0.0

# Calculate electric field contributions from each segment
for theta in theta_values:
    # Electric field contribution from each segment
dq = linear_charge_density * R * (theta_total / num_segments)  # Charge of the segment
    dE = k * dq / R**2  # Electric field due to the segment
    E_total += dE  # Sum contributions

# Print the result with units
print(f'Magnitude of the electric field at the center of curvature (numerical integration): {E_total:.2f} N/C')