# filename: neutron_thermal_wavelength_improved.py

from scipy.constants import h, k, neutron_mass

def calculate_neutron_wavelength(T=373):
    """Calculate the typical thermal wavelength of neutrons at temperature T (in Kelvin).

    Args:
        T (float): Temperature in Kelvin. Must be positive.

    Returns:
        float: Thermal wavelength in meters.
    """
    if T <= 0:
        raise ValueError("Temperature must be positive.")

    wavelength = h / ((neutron_mass * k * T) ** 0.5)
    return wavelength

# Calculate and print the neutron wavelength at 373 K
neutron_wavelength = calculate_neutron_wavelength()
print(f"Typical neutron wavelength at 373 K: {neutron_wavelength:.3e} meters")
