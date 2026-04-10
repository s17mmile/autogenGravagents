import numpy as np

# Constants
hbar = 1.055e-34  # Reduced Planck's constant in Js
Z = 126           # Atomic number for element with atomic number 126
e = 1.602e-19     # Elementary charge in C
alpha = 1/137     # Fine-structure constant
n = 1             # Principal quantum number for innermost electron

def calculate_radius(Z, n):
    """
    Calculate the most probable distance of the innermost electrons from the nucleus
    of a hydrogen-like atom using a modified formula based on the Dirac equation.
    
    Parameters:
    Z (int): Atomic number of the element.
    n (int): Principal quantum number.
    
    Returns:
    float: The radius in meters.
    """
    # Calculate the radius using the modified formula
    r = (n**2 * hbar**2) / (Z * e**2) * (1 / np.sqrt(1 + (Z**2 * alpha**2) / (n**2)))
    return r

# Calculate the radius for atomic number 126
radius = calculate_radius(Z, n)

# Print the result with units
print(f'The most probable distance of the innermost electrons from the nucleus is approximately {radius:.2e} meters.')