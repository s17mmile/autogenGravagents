# filename: electric_field_spherical_shells.py

def electric_field_at_r(r_meters, Q_coulombs):
    """Calculate electric field at distance r from a point charge Q.

    Args:
        r_meters (float): Distance from charge in meters (must be > 0).
        Q_coulombs (float): Charge in coulombs.

    Returns:
        float: Electric field magnitude in N/C.

    Raises:
        ValueError: If r_meters is not positive.
    """
    if r_meters <= 0:
        raise ValueError("Distance r must be positive and non-zero.")
    k = 8.99e9  # Coulomb's constant in N m^2 / C^2
    E = k * Q_coulombs / (r_meters ** 2)
    return E

# Given values
r_cm = 12.0
r_m = r_cm / 100  # convert cm to meters
Q_inner = 4.00e-8  # charge on inner shell in Coulombs

# Calculate electric field at r = 12.0 cm
E_at_12cm = electric_field_at_r(r_m, Q_inner)

# Output the result
print(f"Electric field at r = {r_cm} cm due to inner shell charge: {E_at_12cm:.2e} N/C")
