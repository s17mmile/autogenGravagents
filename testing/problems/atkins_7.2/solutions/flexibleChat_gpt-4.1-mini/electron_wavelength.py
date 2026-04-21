# filename: electron_wavelength.py

from scipy.constants import h, m_e, e

def calculate_electron_wavelength(V_kV=40):
    """Calculate the de Broglie wavelength of an electron accelerated through a potential difference.

    Args:
        V_kV (float): Accelerating voltage in kilovolts (kV). Must be positive.

    Returns:
        float: Wavelength in meters.
    """
    if V_kV <= 0:
        raise ValueError("Voltage must be positive.")

    V = V_kV * 1e3  # Convert kV to V

    # Calculate momentum p = sqrt(2 * m * e * V)
    p = (2 * m_e * e * V) ** 0.5

    # Calculate wavelength lambda = h / p
    wavelength = h / p

    return wavelength

if __name__ == '__main__':
    wavelength_m = calculate_electron_wavelength(40)
    print(f"De Broglie wavelength of electron accelerated through 40 kV: {wavelength_m:.3e} meters")
