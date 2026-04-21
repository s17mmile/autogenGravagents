# filename: force_ratio_electron_positron.py

def calculate_force_ratio():
    """Calculate the ratio of electrical force to gravitational force between an electron and a positron.

    Returns:
        float: The ratio (dimensionless) indicating how much stronger the electrical force is compared to gravity.
    """
    # Constants with units
    k_e = 8.988e9  # Coulomb's constant in N m^2 / C^2
    e = 1.602e-19  # Elementary charge in Coulombs
    G = 6.674e-11  # Gravitational constant in N m^2 / kg^2
    m = 9.11e-31   # Mass of electron/positron in kg

    # Calculate ratio of electrical force to gravitational force
    ratio = (k_e * e**2) / (G * m**2)
    return ratio

if __name__ == '__main__':
    ratio = calculate_force_ratio()
    print(f"Ratio of electrical force to gravitational force between electron and positron: {ratio:.3e} (dimensionless)")
