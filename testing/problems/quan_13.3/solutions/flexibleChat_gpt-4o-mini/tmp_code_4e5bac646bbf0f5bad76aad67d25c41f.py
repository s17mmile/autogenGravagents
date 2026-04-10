def calculate_ionization_energy(D0_H2, D0_H2_plus):
    """
    Calculate the first ionization energy of molecular hydrogen (H2).
    
    Parameters:
    D0_H2 (float): Dissociation energy of H2 in eV.
    D0_H2_plus (float): Dissociation energy of H2+ in eV.
    
    Returns:
    float: The ionization energy of H2 in eV.
    """
    # Check if the inputs are valid numbers
    if not (isinstance(D0_H2, (int, float)) and isinstance(D0_H2_plus, (int, float))):
        raise ValueError('Dissociation energies must be valid numbers.')
    
    # Calculate the first ionization energy of H2
    ionization_energy = D0_H2 - D0_H2_plus
    return ionization_energy

# Constants for dissociation energies in eV
D0_H2 = 4.478  # Dissociation energy of H2
D0_H2_plus = 2.651  # Dissociation energy of H2+

# Calculate the ionization energy
ionization_energy = calculate_ionization_energy(D0_H2, D0_H2_plus)

# Output the result
print(f'The first ionization energy of H2 is {ionization_energy:.3f} eV')