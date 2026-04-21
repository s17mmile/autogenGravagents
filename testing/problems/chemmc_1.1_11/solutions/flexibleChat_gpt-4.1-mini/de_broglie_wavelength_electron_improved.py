# filename: de_broglie_wavelength_electron_improved.py

from scipy.constants import h, m_e, c

def calculate_de_broglie_wavelength(velocity):
    """
    Calculate the de Broglie wavelength of an electron.

    Parameters:
    velocity (float): Electron velocity in meters per second (m/s). Must be positive.

    Returns:
    float: de Broglie wavelength in meters (m).
    """
    if velocity <= 0:
        raise ValueError("Velocity must be positive and non-zero.")

    # Calculate momentum p = m * v
    p = m_e * velocity

    # Calculate de Broglie wavelength lambda = h / p
    wavelength = h / p

    return wavelength

# Electron velocity at 1.00% of speed of light
electron_velocity = 0.01 * c

# Calculate and print the wavelength
wavelength = calculate_de_broglie_wavelength(electron_velocity)
print(f"De Broglie wavelength of the electron: {wavelength:.3e} meters")
