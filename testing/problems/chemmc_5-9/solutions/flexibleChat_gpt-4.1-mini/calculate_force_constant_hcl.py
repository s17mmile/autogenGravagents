# filename: calculate_force_constant_hcl.py
"""
Calculate the force constant k of the HCl molecule from the Morse potential parameters.

The force constant k is given by k = 2 * D * beta^2,
where D is the dissociation energy (Joules per molecule) and beta is a parameter related to the potential well width (inverse meters).
"""

def calculate_force_constant(D, beta):
    """Calculate the force constant k from D and beta."""
    return 2 * D * beta**2


# Given constants
D = 7.31e-19  # Dissociation energy in Joules per molecule
beta = 1.81e10  # Beta parameter in inverse meters

# Calculate force constant k
k = calculate_force_constant(D, beta)

# Print the result
print(f"Force constant k of HCl: {k:.3e} N/m")
