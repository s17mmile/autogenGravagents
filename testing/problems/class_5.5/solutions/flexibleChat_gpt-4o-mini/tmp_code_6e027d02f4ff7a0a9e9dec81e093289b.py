def calculate_max_tidal_change():
    """
    Calculate the maximum height change in ocean tides caused by the Moon.
    This function uses the gravitational force formula to estimate the tidal force
    and provides a rough estimate of the maximum height change in meters.
    """
    # Constants
    G = 6.674 * 10**-11  # Gravitational constant in N(m/kg)^2
    M = 7.35 * 10**22    # Mass of the Moon in kg
    m = 5.97 * 10**24     # Mass of the Earth in kg
    r = 3.84 * 10**8      # Distance from Earth to Moon in meters

    # Calculate tidal force
    T = (2 * G * M * m) / (r ** 2)

    # Estimate maximum height change based on tidal force
    # This is a simplified model and does not account for local geography
    # A more accurate model would require complex hydrodynamic analysis
    max_height_change = T / (m * 9.81)  # Rough estimate of height change in meters

    return max_height_change

# Execute the function and print the result
max_height = calculate_max_tidal_change()
print(f'Maximum height change in ocean tides caused by the Moon: {max_height:.6f} meters')