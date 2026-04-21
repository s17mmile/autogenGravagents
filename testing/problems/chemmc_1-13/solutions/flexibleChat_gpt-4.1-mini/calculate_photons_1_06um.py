# filename: calculate_photons_1_06um.py

from scipy.constants import h, c

def calculate_photons(pulse_energy_mJ, wavelength_um):
    """Calculate the number of photons in a light pulse.

    Args:
        pulse_energy_mJ (float): Pulse energy in millijoules.
        wavelength_um (float): Wavelength in micrometers.

    Returns:
        int: Number of photons (rounded to nearest integer).
    """
    if pulse_energy_mJ <= 0 or wavelength_um <= 0:
        raise ValueError("Pulse energy and wavelength must be positive values.")

    # Convert inputs to SI units
    pulse_energy_J = pulse_energy_mJ * 1e-3  # mJ to J
    wavelength_m = wavelength_um * 1e-6      # micrometers to meters

    # Energy of one photon
    E_photon = h * c / wavelength_m

    # Number of photons
    n_photons = pulse_energy_J / E_photon

    return int(round(n_photons))

# Given values
pulse_energy_mJ = 2.00
wavelength_um = 1.06

# Calculate number of photons
number_of_photons = calculate_photons(pulse_energy_mJ, wavelength_um)

print(f"Number of photons in a 2.00 mJ pulse at 1.06 micrometers: {number_of_photons}")
