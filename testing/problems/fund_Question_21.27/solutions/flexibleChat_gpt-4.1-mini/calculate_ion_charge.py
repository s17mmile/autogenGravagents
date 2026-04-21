# filename: calculate_ion_charge.py

def calculate_ion_charge(F, r, k=8.99e9):
    """
    Calculate the charge of each ion using Coulomb's law.

    Parameters:
    F (float): Electrostatic force in Newtons (must be positive).
    r (float): Distance between ions in meters (must be positive).
    k (float): Coulomb's constant in N m^2 / C^2 (default 8.99e9).

    Returns:
    float: Charge of each ion in Coulombs.
    """
    if F <= 0 or r <= 0:
        raise ValueError("Force and distance must be positive numbers.")
    q = (F * r**2 / k)**0.5
    return q


if __name__ == '__main__':
    # Given values
    F = 3.7e-9  # Force in Newtons
    r = 5.0e-10  # Distance in meters

    charge = calculate_ion_charge(F, r)
    print(f"Charge of each ion: {charge:.3e} Coulombs")
