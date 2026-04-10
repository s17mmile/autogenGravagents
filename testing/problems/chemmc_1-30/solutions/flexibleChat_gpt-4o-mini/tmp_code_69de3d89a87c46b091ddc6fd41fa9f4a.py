def calculate_reduced_mass(atomic_mass):
    """Calculate the reduced mass of a diatomic molecule given the atomic mass of one atom."""
    return atomic_mass / 2

# Constants
atomic_mass_nitrogen = 14.00  # atomic mass of nitrogen in atomic mass units (u)

# Calculate reduced mass for nitrogen molecule (N2)
reduced_mass = calculate_reduced_mass(atomic_mass_nitrogen)

# Output the result
print(f'The reduced mass of a nitrogen molecule (N2) is: {reduced_mass} atomic mass units (u)')