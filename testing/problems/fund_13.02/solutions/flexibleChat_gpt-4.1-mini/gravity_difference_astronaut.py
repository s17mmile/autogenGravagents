# filename: gravity_difference_astronaut.py

def gravity_difference(h=1.70, r=6.77e6):
    """
    Calculate the difference in gravitational acceleration between the feet and head of an astronaut.

    Parameters:
    h (float): Height of the astronaut in meters (default 1.70 m).
    r (float): Distance from the Earth's center to the astronaut's feet in meters (default 6.77e6 m).

    Returns:
    float: Difference in gravitational acceleration (m/s^2) between feet and head.

    Note:
    The height h should be much smaller than r for the approximation to hold.
    """
    if h <= 0 or r <= 0:
        raise ValueError("Height and radius must be positive numbers.")

    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    M = 5.972e24     # mass of Earth in kg

    # Calculate gravitational acceleration at feet
    g_feet = G * M / r**2

    # Calculate gravitational acceleration at head
    g_head = G * M / (r + h)**2

    # Difference in gravitational acceleration
    delta_g = g_feet - g_head

    return delta_g

# Calculate and print the difference
difference = gravity_difference()
print(f"Difference in gravitational acceleration between feet and head: {difference:.6e} m/s^2")
