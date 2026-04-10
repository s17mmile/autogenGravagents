import math

# Constants
hbar = 1.055e-34  # Reduced Planck's constant in J.s
m = 9.11e-31      # Mass of electron in kg

# Given values
V0 = 20.0 * 1.602e-19  # Depth of the well in Joules
E1 = 3.00 * 1.602e-19   # Energy of the lowest state in Joules

# Calculate the width of the well using the formula
L = math.sqrt((1**2 * hbar**2 * math.pi**2) / (2 * m * E1))

# Output the width in meters
print(f'The width of the well is: {L} meters')