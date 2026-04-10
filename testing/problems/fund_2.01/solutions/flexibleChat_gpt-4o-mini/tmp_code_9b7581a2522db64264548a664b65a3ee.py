def calculate_displacement(driving_distance, walking_distance):
    return driving_distance + walking_distance

# Constants for distances
driving_distance_km = 8.4  # Distance driven in kilometers
walking_distance_km = 2.0   # Distance walked in kilometers

# Calculate total displacement
 total_displacement_km = calculate_displacement(driving_distance_km, walking_distance_km)

# Output the result
print(f'Total Displacement: {total_displacement_km} km')