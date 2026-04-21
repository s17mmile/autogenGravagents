# filename: calculate_nacl_dipole.py

def calculate_dipole_moment(I_Na_eV, EA_Cl_eV, R_e_A):
    """
    Calculate the dipole moment of NaCl assuming full and partial ionic charge transfer.

    Parameters:
    I_Na_eV (float): Ionization energy of Na in electron volts.
    EA_Cl_eV (float): Electron affinity of Cl in electron volts.
    R_e_A (float): Equilibrium distance between ions in Angstroms.

    Returns:
    tuple: Dipole moments (Debye) assuming full charge transfer and partial charge transfer.

    The partial charge transfer fraction is estimated as (I_Na - EA_Cl) / (I_Na + EA_Cl),
    reflecting the degree of ionic character in the bond.
    """
    # Constants
    e_charge = 1.602e-19  # elementary charge in Coulombs
    angstrom_to_m = 1e-10  # conversion from Angstrom to meters
    debye_to_Cm = 3.33564e-30  # 1 Debye in C*m

    # Input validation
    if I_Na_eV <= 0 or EA_Cl_eV <= 0:
        raise ValueError("Ionization energy and electron affinity must be positive.")
    if (I_Na_eV + EA_Cl_eV) == 0:
        raise ValueError("Sum of ionization energy and electron affinity must not be zero.")

    # Convert equilibrium distance to meters
    R_e_m = R_e_A * angstrom_to_m

    # Full ionic charge assumption (q = elementary charge)
    q_full = e_charge
    D_full_Cm = q_full * R_e_m
    D_full_Debye = D_full_Cm / debye_to_Cm

    # Partial charge transfer estimate
    q_eff_fraction = (I_Na_eV - EA_Cl_eV) / (I_Na_eV + EA_Cl_eV)
    # Ensure charge fraction is non-negative
    q_eff_fraction = max(q_eff_fraction, 0)
    q_eff = q_eff_fraction * e_charge
    D_partial_Cm = q_eff * R_e_m
    D_partial_Debye = D_partial_Cm / debye_to_Cm

    return D_full_Debye, D_partial_Debye


if __name__ == '__main__':
    # Given values
    I_Na = 5.14  # eV
    EA_Cl = 3.61  # eV
    R_e = 2.36  # Angstrom

    D_full, D_partial = calculate_dipole_moment(I_Na, EA_Cl, R_e)

    print(f"Dipole moment assuming full ionic charge transfer: {D_full:.2f} Debye")
    print(f"Dipole moment with partial charge transfer estimate: {D_partial:.2f} Debye")
