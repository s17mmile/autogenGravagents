def calculate_force_constant(D, beta):
    """
    Calculate the force constant k for a given Morse potential.
    Parameters:
    D (float): Depth of the potential well in J per molecule.
    beta (float): Parameter related to the width of the potential in m^-1.
    Returns:
    float: The calculated force constant in N/m.
    """
    # Calculate beta squared
    beta_squared = beta ** 2
    # Calculate the force constant k
    k = D * beta_squared
    return k

# Constants for HCl
D = 7.31e-19  # J per molecule
beta = 1.81e10  # m^-1

# Calculate the force constant
force_constant = calculate_force_constant(D, beta)

# Output the result
print(f'The force constant k for HCl is approximately {force_constant:.2f} N/m')