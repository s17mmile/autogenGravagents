import math

def calculate_charge(F, r):
    # Coulomb's constant in N m^2/C^2
    coulombs_constant = 8.99e9  
    # Calculate r squared
    r_squared = r ** 2
    # Calculate charge using rearranged Coulomb's Law
    q = math.sqrt((F * r_squared) / coulombs_constant)
    return q

# Given values
F = 3.7e-9  # Electrostatic force in Newtons
r = 5.0e-10  # Distance in meters

# Validate inputs
if F <= 0 or r <= 0:
    raise ValueError('Force and distance must be positive values.')

# Calculate charge
charge = calculate_charge(F, r)

# Print the charge of each ion
print(f'The charge of each ion is approximately {charge:.2e} C')