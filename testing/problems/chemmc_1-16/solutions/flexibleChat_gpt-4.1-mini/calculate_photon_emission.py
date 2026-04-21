# filename: calculate_photon_emission.py

def photons_per_second(power_mw=1.00, wavelength_nm=337):
    """
    Calculate the number of photons emitted per second by a laser.

    Parameters:
    power_mw (float): Power of the laser in milliwatts (mW). Must be positive.
    wavelength_nm (float): Wavelength of the emitted light in nanometers (nm). Must be positive.

    Returns:
    float: Number of photons emitted per second.

    Raises:
    ValueError: If power_mw or wavelength_nm is not positive.
    """
    if power_mw <= 0:
        raise ValueError("Power must be a positive number.")
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be a positive number.")

    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    c = 3.00e8     # Speed of light in m/s

    # Convert power from mW to W
    power_w = power_mw * 1e-3

    # Convert wavelength from nm to m
    wavelength_m = wavelength_nm * 1e-9

    # Calculate energy per photon (E = hc / lambda)
    energy_per_photon = h * c / wavelength_m

    # Calculate number of photons per second (N = P / E)
    photons_per_sec = power_w / energy_per_photon

    return photons_per_sec

# Example usage
photon_count = photons_per_second()
print(f'Number of photons emitted per second: {photon_count:.2e}')
