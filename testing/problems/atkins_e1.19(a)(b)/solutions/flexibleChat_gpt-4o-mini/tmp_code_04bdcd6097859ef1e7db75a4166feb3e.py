import math

# Constants
AVOGADRO_NUMBER = 6.022e23  # molecules/mol
CM3_TO_M3 = 1e-6  # conversion factor from cm^3 to m^3
ANGSTROMS_TO_METERS = 1e-10  # conversion factor from Å to m

def estimate_methane_radius():
    """
    Estimate the radius of a methane molecule using its critical constants.
    Returns:
        float: Estimated radius in angstroms.
    """
    # Given critical constants for methane
    p_c = 45.6  # atm
    V_c = 98.7 * CM3_TO_M3  # m^3/mol (converted from cm^3/mol)
    T_c = 190.6  # K

    # Step 1: Calculate the volume occupied by a single molecule
    V_molecule = V_c / AVOGADRO_NUMBER  # m^3

    # Step 2: Estimate the radius of the molecule assuming spherical shape
    if V_molecule <= 0:
        raise ValueError('Volume per molecule must be positive.')
    r = (3 * V_molecule / (4 * math.pi)) ** (1/3)  # m

    # Step 3: Convert radius to angstroms
    r_angstroms = r / ANGSTROMS_TO_METERS  # Å
    return r_angstroms

# Output the estimated radius
estimated_radius = estimate_methane_radius()
print(f'Estimated radius of a methane molecule: {estimated_radius:.2f} Å')