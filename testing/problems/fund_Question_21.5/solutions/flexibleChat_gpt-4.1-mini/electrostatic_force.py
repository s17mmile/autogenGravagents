# filename: electrostatic_force.py

def calculate_electrostatic_force(q1, q2, r):
    """Calculate the magnitude of the electrostatic force between two charges.

    Args:
        q1 (float): Charge of the first particle in Coulombs.
        q2 (float): Charge of the second particle in Coulombs.
        r (float): Distance between the charges in meters.

    Returns:
        float: Magnitude of the electrostatic force in Newtons.

    Raises:
        ValueError: If the distance r is zero or negative.
    """
    if r <= 0:
        raise ValueError("Distance between charges must be greater than zero.")
    k = 8.988e9  # Coulomb's constant in N m^2 / C^2
    force = k * abs(q1 * q2) / (r ** 2)
    return force

# Given values
q1 = 3.00e-6  # charge in Coulombs
q2 = -1.50e-6  # charge in Coulombs
r = 0.12  # distance in meters

force_magnitude = calculate_electrostatic_force(q1, q2, r)
print(f"The magnitude of the electrostatic force is {force_magnitude:.3e} N")
