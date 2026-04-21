# filename: electron_stop_distance.py

def calculate_launch_distance(sigma, kinetic_energy, electron_charge=1.602176634e-19, epsilon_0=8.854187817e-12):
    """
    Calculate the distance from the charged plate where the electron must be launched to stop at the plate.

    Parameters:
    sigma (float): Surface charge density in C/m^2 (negative value for negative charge)
    kinetic_energy (float): Initial kinetic energy of the electron in Joules
    electron_charge (float): Magnitude of the electron charge in Coulombs (default 1.602176634e-19 C)
    epsilon_0 (float): Permittivity of free space in C^2/(N m^2) (default 8.854187817e-12)

    Returns:
    float: Distance in meters
    """
    if sigma == 0:
        raise ValueError("Surface charge density sigma must be non-zero.")
    if kinetic_energy <= 0:
        raise ValueError("Kinetic energy must be positive.")

    distance = (2 * epsilon_0 * kinetic_energy) / (electron_charge * abs(sigma))
    return distance

# Given values
sigma = -2.0e-6  # C/m^2
kinetic_energy = 1.60e-17  # J

# Calculate distance
distance_m = calculate_launch_distance(sigma, kinetic_energy)

# Print result with units
print(f"Distance from plate: {distance_m:.6f} meters")
print(f"Distance from plate: {distance_m * 1000:.3f} millimeters")
