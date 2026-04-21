# filename: calculate_charge.py

def calculate_charge(E, r):
    """Calculate the magnitude of charge producing an electric field.

    Parameters:
    E (float): Electric field magnitude in N/C
    r (float): Distance from the charge in meters

    Returns:
    float: Charge magnitude in coulombs
    """
    # Coulomb's constant in N m^2 / C^2
    k = 8.99e9
    if E <= 0 or r <= 0:
        raise ValueError("Electric field and distance must be positive numbers.")
    # Calculate charge magnitude
    q = E * r**2 / k
    return q

# Given values
E = 2.0  # Electric field in N/C
r = 0.50  # Distance in meters

charge = calculate_charge(E, r)
print(f"The magnitude of the charge is {charge:.3e} coulombs.")
