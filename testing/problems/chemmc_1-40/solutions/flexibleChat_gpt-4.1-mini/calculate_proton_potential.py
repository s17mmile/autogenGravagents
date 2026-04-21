# filename: calculate_proton_potential.py

from scipy.constants import h, m_p, e

def calculate_potential_for_deBroglie_wavelength(wavelength_meters):
    """
    Calculate the potential difference (in volts) through which a proton must fall
    to have a given de Broglie wavelength.

    Parameters:
    wavelength_meters (float): The de Broglie wavelength in meters (must be positive).

    Returns:
    float: The potential difference in volts.
    """
    if wavelength_meters <= 0:
        raise ValueError("Wavelength must be positive and non-zero.")

    V = (h ** 2) / (2 * m_p * e * wavelength_meters ** 2)
    return V

# Given de Broglie wavelength
wavelength = 1.0e-10  # meters

potential_difference = calculate_potential_for_deBroglie_wavelength(wavelength)

print(f"Potential difference V (in volts) for proton to have de Broglie wavelength {wavelength} m: {potential_difference:.2f} V")
