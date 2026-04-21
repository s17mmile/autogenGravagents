# filename: entropy_change_nitrogen.py
import math

def calculate_entropy_change(mass_g, molar_mass_g_per_mol):
    """Calculate the entropy change for an isothermal expansion where volume doubles.

    Args:
        mass_g (float): Mass of the gas in grams (must be positive).
        molar_mass_g_per_mol (float): Molar mass of the gas in g/mol (must be positive).

    Returns:
        float: Entropy change in J/K.

    Raises:
        ValueError: If mass_g or molar_mass_g_per_mol is not positive.
    """
    if mass_g <= 0 or molar_mass_g_per_mol <= 0:
        raise ValueError("Mass and molar mass must be positive numbers.")

    R = 8.314  # Gas constant in J/(mol*K)
    n = mass_g / molar_mass_g_per_mol  # number of moles
    delta_S = n * R * math.log(2)  # entropy change for doubling volume
    return delta_S

# Given data
mass_nitrogen = 14.0  # grams
molar_mass_nitrogen = 28.02  # g/mol

# Calculate entropy change
entropy_change = calculate_entropy_change(mass_nitrogen, molar_mass_nitrogen)

# Save result to a file
with open('entropy_change_result.txt', 'w') as f:
    f.write(f'Total entropy change (J/K): {entropy_change:.4f}\n')
