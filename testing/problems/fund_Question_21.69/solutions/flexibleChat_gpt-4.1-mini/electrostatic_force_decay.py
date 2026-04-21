# filename: electrostatic_force_decay.py

def calculate_electrostatic_force(q1=2*1.602e-19, q2=90*1.602e-19, r=9.0e-15):
    """Calculate the magnitude of the electrostatic force between two point charges.

    Args:
        q1 (float): Charge of first particle in Coulombs.
        q2 (float): Charge of second particle in Coulombs.
        r (float): Separation distance in meters.

    Returns:
        float: Electrostatic force magnitude in Newtons.
    """
    k = 8.988e9  # Coulomb's constant in N m^2 / C^2
    force = k * (q1 * q2) / (r ** 2)
    return force

if __name__ == '__main__':
    force = calculate_electrostatic_force()
    print(f"Electrostatic force magnitude: {force:.3e} N")
