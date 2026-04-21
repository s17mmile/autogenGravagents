# filename: calculate_benzene_bond_energy_improved.py

def calculate_bond_energy():
    """
    Calculate the total energy needed to compress three carbon-carbon single bonds
    and stretch three carbon-carbon double bonds to the benzene bond length using
    the harmonic oscillator potential energy formula.

    Returns:
        tuple: Total energy in Joules and electronvolts (eV)
    """
    # Constants: bond lengths in Angstroms
    single_bond_length_A = 1.53  # Typical C-C single bond length
    double_bond_length_A = 1.335  # Typical C=C double bond length
    benzene_bond_length_A = 1.397  # Benzene bond length

    # Force constants in N/m
    k_single = 500
    k_double = 950

    # Number of bonds
    n_single = 3
    n_double = 3

    # Conversion factor from Angstrom to meters
    angstrom_to_m = 1e-10

    # Convert bond lengths to meters
    single_bond_length = single_bond_length_A * angstrom_to_m
    double_bond_length = double_bond_length_A * angstrom_to_m
    benzene_bond_length = benzene_bond_length_A * angstrom_to_m

    # Calculate displacement (delta x) for single and double bonds
    delta_x_single = benzene_bond_length - single_bond_length  # Compression (negative value)
    delta_x_double = benzene_bond_length - double_bond_length  # Stretching (positive value)

    # Calculate energy for one bond using harmonic oscillator potential: E = 0.5 * k * (delta_x)^2
    E_single = 0.5 * k_single * delta_x_single**2
    E_double = 0.5 * k_double * delta_x_double**2

    # Total energy for all bonds
    E_total = n_single * E_single + n_double * E_double

    # Convert energy from Joules to electronvolts (1 eV = 1.60218e-19 J)
    joule_to_eV = 1 / 1.60218e-19
    E_total_eV = E_total * joule_to_eV

    # Print results
    print(f"Total energy needed to compress and stretch bonds: {E_total:.3e} J")
    print(f"Total energy in electronvolts (eV): {E_total_eV:.3e} eV")

    return E_total, E_total_eV

# Execute the calculation
calculate_bond_energy()
