# filename: calculate_slit_distance.py

def calculate_slit_distance(wavelength_nm, screen_distance_m, fringe_spacing_cm):
    """Calculate the distance between two slits in meters.

    Args:
        wavelength_nm (float): Wavelength of light in nanometers.
        screen_distance_m (float): Distance from slits to screen in meters.
        fringe_spacing_cm (float): Distance between bright fringes in centimeters.

    Returns:
        float: Distance between the slits in meters.
    """
    # Input validation
    if wavelength_nm <= 0 or screen_distance_m <= 0 or fringe_spacing_cm <= 0:
        raise ValueError("All input values must be positive numbers.")

    # Convert units to meters
    wavelength_m = wavelength_nm * 1e-9  # nanometers to meters
    fringe_spacing_m = fringe_spacing_cm * 1e-2  # centimeters to meters

    # Calculate slit distance using d = (wavelength * L) / fringe_spacing
    slit_distance_m = (wavelength_m * screen_distance_m) / fringe_spacing_m
    return slit_distance_m


# Given values
wavelength_nm = 694.3  # wavelength in nanometers
screen_distance_m = 3.00  # distance to screen in meters
fringe_spacing_cm = 1.50  # fringe spacing in centimeters

# Calculate slit distance
slit_distance = calculate_slit_distance(wavelength_nm, screen_distance_m, fringe_spacing_cm)

# Print the result
print(f"Distance between the slits: {slit_distance:.6e} meters")
print(f"Distance between the slits: {slit_distance * 1e3:.3f} millimeters")
