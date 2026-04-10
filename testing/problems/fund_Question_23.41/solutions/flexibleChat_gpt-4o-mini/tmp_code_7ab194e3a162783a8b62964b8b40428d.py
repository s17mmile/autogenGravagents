import numpy as np

# Constants
sigma = -2.0e-6  # Surface charge density in C/m^2
KE = 1.60e-17    # Initial kinetic energy in J
q = 1.6e-19      # Charge of the electron in C
epsilon_0 = 8.85e-12  # Permittivity of free space in C^2/(N*m^2)

def calculate_distance(sigma, KE, q, epsilon_0):
    # Step 1: Calculate the electric field (E) due to the plate
    E = abs(sigma / (2 * epsilon_0))  # Electric field in N/C
    # Step 2: Relate kinetic energy to work done by electric field
    # KE = q * E * d => d = KE / (q * E)
    # Step 3: Solve for d
    d = KE / (q * E)  # Distance in meters
    return d

# Calculate the distance
distance = calculate_distance(sigma, KE, q, epsilon_0)

# Output the result
print(f'Distance from the plate: {distance:.3f} meters')