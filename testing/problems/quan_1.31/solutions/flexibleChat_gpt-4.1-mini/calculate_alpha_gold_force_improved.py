# filename: calculate_alpha_gold_force_improved.py

from scipy.constants import k, e

def calculate_force_on_alpha_particle(distance_angstrom):
    """
    Calculate the electrostatic force on an alpha particle passing near a gold nucleus.

    Parameters:
    distance_angstrom (float): Distance between alpha particle and gold nucleus in angstroms.

    Returns:
    float: Force in newtons.

    Raises:
    ValueError: If distance is not positive.
    """
    if distance_angstrom <= 0:
        raise ValueError("Distance must be positive and non-zero.")

    # Charges
    q_alpha = 2 * e  # Alpha particle charge (+2e)
    q_gold = 79 * e  # Gold nucleus charge (+79e)

    # Convert distance from angstroms to meters
    distance_m = distance_angstrom * 1e-10

    # Calculate force using Coulomb's law
    force = k * abs(q_alpha * q_gold) / (distance_m ** 2)

    return force

# Given distance
try:
    distance_angstrom = 0.00300
    force_newtons = calculate_force_on_alpha_particle(distance_angstrom)
    print(f'Force on alpha particle at {distance_angstrom} Angstroms from gold nucleus: {force_newtons:.3e} N')
    with open('force_result.txt', 'w') as f:
        f.write(f'Force on alpha particle at {distance_angstrom} Angstroms from gold nucleus: {force_newtons:.3e} N\n')
except ValueError as ve:
    print(f'Error: {ve}')
