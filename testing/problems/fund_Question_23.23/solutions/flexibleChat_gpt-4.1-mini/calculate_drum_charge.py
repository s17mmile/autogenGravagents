# filename: calculate_drum_charge.py

def calculate_total_charge(length_cm, diameter_cm, electric_field):
    """Calculate the total charge on a cylindrical drum given its dimensions and electric field.

    Parameters:
    length_cm (float): Length of the drum in centimeters.
    diameter_cm (float): Diameter of the drum in centimeters.
    electric_field (float): Electric field just above the drum surface in N/C.

    Returns:
    float: Total charge on the drum in Coulombs.
    """
    import math

    # Constants
    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(N m^2)

    # Input validation
    if length_cm <= 0 or diameter_cm <= 0 or electric_field <= 0:
        raise ValueError("Length, diameter, and electric field must be positive numbers.")

    # Convert dimensions to meters
    length_m = length_cm / 100
    radius_m = diameter_cm / 2 / 100

    # Calculate surface charge density sigma = E * epsilon_0
    sigma = electric_field * epsilon_0

    # Calculate lateral surface area of the cylinder A = 2 * pi * r * L
    surface_area = 2 * math.pi * radius_m * length_m

    # Calculate total charge Q = sigma * A
    total_charge = sigma * surface_area

    return total_charge

# Given data
length_cm = 42
diameter_cm = 12
electric_field = 2.3e5  # N/C

# Calculate total charge
charge = calculate_total_charge(length_cm, diameter_cm, electric_field)

# Print the result
print(f"Total charge on the drum: {charge:.4e} Coulombs")
