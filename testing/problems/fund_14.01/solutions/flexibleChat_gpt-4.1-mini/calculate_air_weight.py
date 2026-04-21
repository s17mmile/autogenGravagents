# filename: calculate_air_weight.py

def calculate_air_weight(length_m, width_m, height_m, air_density_kg_per_m3=1.2, gravity_m_per_s2=9.81):
    """Calculate the weight of air in a room given dimensions and air density.

    Parameters:
    length_m (float): Length of the room in meters.
    width_m (float): Width of the room in meters.
    height_m (float): Height of the room in meters.
    air_density_kg_per_m3 (float): Density of air in kg/m^3. Default is 1.2 (room temperature).
    gravity_m_per_s2 (float): Gravitational acceleration in m/s^2. Default is 9.81.

    Returns:
    float: Weight of the air in newtons.
    """
    if length_m <= 0 or width_m <= 0 or height_m <= 0:
        raise ValueError("Room dimensions must be positive numbers.")
    if air_density_kg_per_m3 <= 0 or gravity_m_per_s2 <= 0:
        raise ValueError("Air density and gravity must be positive numbers.")

    volume_m3 = length_m * width_m * height_m
    mass_kg = air_density_kg_per_m3 * volume_m3
    weight_newtons = mass_kg * gravity_m_per_s2
    return weight_newtons

# Given room dimensions
length = 3.5  # meters
width = 4.2   # meters
height = 2.4  # meters

# Calculate air weight
weight = calculate_air_weight(length, width, height)

# Save the result to a file with error handling
try:
    with open('air_weight_result.txt', 'w') as f:
        f.write(f'Weight of air in the room: {weight:.2f} newtons\n')
except IOError as e:
    print(f'Error writing to file: {e}')
