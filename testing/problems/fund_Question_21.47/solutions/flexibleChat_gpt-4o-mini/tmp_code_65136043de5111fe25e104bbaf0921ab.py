import numpy as np

# Constants
k = 8.99e9  # Coulomb's constant in N m^2/C^2
q1 = 6.0e-6  # Charge at x=8.0 m in C
q2 = -4.0e-6  # Charge at x=16.0 m in C
x1 = 8.0  # Position of charge 1 in m
x2 = 16.0  # Position of charge 2 in m
x3 = 24.0  # Position of charge 3 in m

# Function to calculate force on charge at origin due to a charge at position x
# Returns the magnitude and direction of the force

def force_on_origin(q, x):
    r = abs(x)  # Distance from origin
    force = k * q / (r ** 2)  # Calculate force magnitude
    return force * (1 if q > 0 else -1)  # Return force with direction

# Calculate forces on a charge at the origin
F1 = force_on_origin(q1, x1)  # Force due to charge 1
F2 = force_on_origin(q2, x2)  # Force due to charge 2

# Set up the equation for net force to be zero
# F1 - F2 + F3 = 0  => F3 = F2 - F1
F3 = F2 - F1

# Calculate the required charge at x=24 m
# F3 = k * |q| / (x3^2) => |q| = F3 * (x3^2) / k
required_charge = F3 * (x3 ** 2) / k

# Output the required charge
if required_charge < 0:
    print(f'Required charge at x=24 m: {abs(required_charge):.2e} C (negative charge)')
else:
    print(f'Required charge at x=24 m: {required_charge:.2e} C (positive charge)')