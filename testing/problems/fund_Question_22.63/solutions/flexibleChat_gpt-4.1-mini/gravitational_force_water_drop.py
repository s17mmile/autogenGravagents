# filename: gravitational_force_water_drop.py
import math

def gravitational_force_on_drop(diameter_micrometers):
    """Calculate gravitational force on a spherical water drop given its diameter in micrometers."""
    # Constants
    density_water = 1000  # kg/m^3
    g = 9.81  # m/s^2

    if diameter_micrometers <= 0:
        raise ValueError("Diameter must be positive")

    # Convert diameter from micrometers to meters
    diameter_m = diameter_micrometers * 1e-6
    radius_m = diameter_m / 2

    # Calculate volume of the sphere
    volume_m3 = (4/3) * math.pi * radius_m**3

    # Calculate mass
    mass_kg = density_water * volume_m3

    # Calculate gravitational force
    F_gravity = mass_kg * g

    return F_gravity

# Given diameter
diameter = 1.20  # micrometers

# Calculate gravitational force
force = gravitational_force_on_drop(diameter)

# Print the result in Newtons
print(f"Gravitational force on the water drop: {force:.3e} N")
