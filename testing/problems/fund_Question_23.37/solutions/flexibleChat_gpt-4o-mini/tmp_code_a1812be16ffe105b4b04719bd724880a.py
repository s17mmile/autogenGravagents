import math

# Constants
Q = 6.0e-6  # Total charge in Coulombs
a = 0.08  # Edge length of the square plate in meters (8.0 cm)
z = 0.0005  # Distance from the center in meters (0.50 mm)
epsilon_0 = 8.85e-12  # Permittivity of free space in C^2/(N m^2)

# Electric field calculation using the formula for a finite square plate
E = (1 / (4 * math.pi * epsilon_0)) * (2 * Q / a**2) * (1 - z / math.sqrt(z**2 + (a / 2)**2))

# Output the result
print(f'The electric field just off the center of the plate is: {E} N/C')