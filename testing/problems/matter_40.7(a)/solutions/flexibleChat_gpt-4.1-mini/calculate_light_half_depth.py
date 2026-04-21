# filename: calculate_light_half_depth.py

def calculate_half_intensity_depth(molar_abs_coeff=6.2e-3, concentration=55.5):
    """
    Calculate the depth at which light intensity is half the surface intensity in seawater.

    Parameters:
    molar_abs_coeff (float): Mean molar absorption coefficient in dm^3 mol^-1 cm^-1.
    concentration (float): Molar concentration in mol dm^-3 (approximate for seawater).

    Returns:
    float: Depth in meters where light intensity is half the surface intensity.
    """
    import math

    # Calculate absorption coefficient k in cm^-1
    k = 2.303 * molar_abs_coeff * concentration

    # Calculate depth x in cm where intensity is half
    x_cm = math.log(2) / k

    # Convert depth to meters
    x_m = x_cm / 100

    return x_m


if __name__ == '__main__':
    depth = calculate_half_intensity_depth()
    print(f"Depth at which light intensity is half the surface intensity: {depth:.4f} meters")
