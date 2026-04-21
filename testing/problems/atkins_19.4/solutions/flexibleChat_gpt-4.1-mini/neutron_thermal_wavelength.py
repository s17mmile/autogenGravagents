# filename: neutron_thermal_wavelength.py

from scipy.constants import h, m_n, k

def calculate_neutron_wavelength(T=373):
    """
    Calculate the typical de Broglie wavelength of neutrons at thermal equilibrium.

    Parameters:
    T (float): Temperature in Kelvin. Must be positive.

    Returns:
    float: Wavelength in meters.
    """
    if T <= 0:
        raise ValueError("Temperature must be a positive number.")

    wavelength = h / ((3 * m_n * k * T) ** 0.5)
    return wavelength

# Calculate and print the wavelength for T=373 K
wavelength = calculate_neutron_wavelength(373)
print(f"Typical neutron wavelength at 373 K: {wavelength:.3e} meters")
