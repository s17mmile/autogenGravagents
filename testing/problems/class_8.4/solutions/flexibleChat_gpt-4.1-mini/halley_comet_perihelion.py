# filename: halley_comet_perihelion.py

def calculate_perihelion_distance(orbital_period_years, eccentricity):
    """
    Calculate the perihelion distance (minimum distance from the Sun) of a comet.
    Parameters:
        orbital_period_years (float): Orbital period in years (must be positive).
        eccentricity (float): Orbital eccentricity (0 <= e < 1).
    Returns:
        float: Perihelion distance in astronomical units (AU).
    """
    # Validate inputs
    if not (0 <= eccentricity < 1):
        raise ValueError("Eccentricity must be between 0 (inclusive) and 1 (exclusive).")
    if orbital_period_years <= 0:
        raise ValueError("Orbital period must be positive.")

    # Kepler's Third Law: T^2 = a^3 (T in years, a in AU)
    # Calculate semi-major axis 'a'
    semi_major_axis = orbital_period_years ** (2/3)

    # Calculate perihelion distance: r_perihelion = a * (1 - e)
    perihelion_distance = semi_major_axis * (1 - eccentricity)

    return perihelion_distance


# Given values for Halley's comet
orbital_period = 76  # years
eccentricity = 0.967

perihelion = calculate_perihelion_distance(orbital_period, eccentricity)

print(f"Halley's comet perihelion distance: {perihelion:.3f} AU")
