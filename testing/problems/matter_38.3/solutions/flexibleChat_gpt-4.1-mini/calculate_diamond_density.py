# filename: calculate_diamond_density.py

import math

def calculate_diamond_density(bond_length_pm=154.45):
    """
    Calculate the expected density of diamond modeled as a close-packed FCC lattice
    of hard spheres with radii equal to half the carbon-carbon bond length.

    Parameters:
    bond_length_pm (float): Carbon-carbon bond length in picometers.

    Returns:
    dict: Dictionary containing lattice parameter (pm and cm), unit cell volume (cm^3),
          mass of atoms in unit cell (g), and calculated density (g/cm^3).
    """
    # Radius of hard sphere in pm
    radius_pm = bond_length_pm / 2  # pm

    # Calculate lattice parameter a in pm using FCC geometry
    a_pm = (4 * radius_pm) / math.sqrt(2)  # pm

    # Convert lattice parameter to cm (1 pm = 1e-10 cm)
    a_cm = a_pm * 1e-10  # cm

    # Volume of unit cell in cm^3
    V_cell_cm3 = a_cm ** 3  # cm^3

    # Number of atoms per FCC unit cell
    n_atoms = 4

    # Atomic mass of carbon in g/mol
    M_C = 12.01  # g/mol

    # Avogadro's number
    N_A = 6.022e23  # atoms/mol

    # Mass of atoms in unit cell in grams
    mass_unit_cell_g = (n_atoms * M_C) / N_A  # g

    # Density calculation in g/cm^3
    density = mass_unit_cell_g / V_cell_cm3  # g/cm^3

    return {
        'lattice_parameter_pm': a_pm,
        'lattice_parameter_cm': a_cm,
        'unit_cell_volume_cm3': V_cell_cm3,
        'mass_unit_cell_g': mass_unit_cell_g,
        'calculated_density_g_cm3': density
    }


if __name__ == '__main__':
    results = calculate_diamond_density()
    actual_density = 3.516  # g/cm^3

    print(f"Lattice parameter a: {results['lattice_parameter_pm']:.4f} pm ({results['lattice_parameter_cm']:.4e} cm)")
    print(f"Unit cell volume: {results['unit_cell_volume_cm3']:.4e} cm^3")
    print(f"Mass of atoms in unit cell: {results['mass_unit_cell_g']:.4e} g")
    print(f"Calculated density: {results['calculated_density_g_cm3']:.4f} g/cm^3")
    print(f"Actual density of diamond: {actual_density} g/cm^3")
