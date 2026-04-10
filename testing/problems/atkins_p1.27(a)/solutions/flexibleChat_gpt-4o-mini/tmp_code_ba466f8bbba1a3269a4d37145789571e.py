def calculate_pressure_difference(density, gravity, height):
    """Calculate the pressure difference in Pascals based on density, gravity, and height."""
    return density * gravity * height

# Constants
DENSITY_AIR = 1.225  # kg/m^3 (density of air at sea level)
GRAVITY = 9.81  # m/s^2 (acceleration due to gravity)
HEIGHT_OF_VESSEL = 0.15  # m (height of the vessel)

# Calculate pressure difference
pressure_difference = calculate_pressure_difference(DENSITY_AIR, GRAVITY, HEIGHT_OF_VESSEL)

# Print the result
print(f'Pressure difference: {pressure_difference:.2f} Pascals')