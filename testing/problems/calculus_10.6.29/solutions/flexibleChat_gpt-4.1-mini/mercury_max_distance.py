# filename: mercury_max_distance.py

def calculate_max_distance(eccentricity, min_distance_km):
    """Calculate the maximum distance (aphelion) of a planet from the sun given its orbital eccentricity and minimum distance (perihelion).

    Parameters:
    eccentricity (float): Orbital eccentricity, must be between 0 and 1.
    min_distance_km (float): Minimum distance from the sun in kilometers, must be positive.

    Returns:
    float: Maximum distance from the sun in kilometers.
    """
    if not (0 <= eccentricity < 1):
        raise ValueError("Eccentricity must be between 0 and 1.")
    if min_distance_km <= 0:
        raise ValueError("Minimum distance must be positive.")

    # Calculate semi-major axis
    a = min_distance_km / (1 - eccentricity)
    # Calculate maximum distance (aphelion)
    max_distance = a * (1 + eccentricity)
    return max_distance

# Given values
eccentricity = 0.206
min_distance_km = 4.6e7  # 4.6 times 10^7 km

max_distance_km = calculate_max_distance(eccentricity, min_distance_km)
print(f"Maximum distance of Mercury from the sun: {max_distance_km:.2f} km")
