# filename: calculate_molar_mass.py

def calculate_molar_mass(a_nm, density_g_cm3, Z):
    """
    Calculate molar mass from crystallographic data.

    Parameters:
    a_nm (float): Unit cell edge length in nanometers.
    density_g_cm3 (float): Density in grams per cubic centimeter.
    Z (int): Number of molecules per unit cell.

    Returns:
    float: Molar mass in grams per mole.
    """
    N_A = 6.022e23  # Avogadro's number in mol^-1

    # Convert edge length from nanometers to centimeters (1 nm = 1e-7 cm)
    a_cm = a_nm * 1e-7  # cm

    # Calculate unit cell volume in cubic centimeters
    V_cell_cm3 = a_cm ** 3  # cm^3

    # Calculate molar mass using the formula M = (density * N_A * V_cell) / Z
    molar_mass = (density_g_cm3 * N_A * V_cell_cm3) / Z  # g/mol

    return molar_mass


if __name__ == '__main__':
    # Given data for tobacco seed globulin
    a_nm = 12.3  # nm
    density = 1.287  # g/cm^3
    Z = 4  # FCC lattice

    molar_mass = calculate_molar_mass(a_nm, density, Z)
    print(f'Molar mass of tobacco seed globulin: {molar_mass:.2f} g/mol')
