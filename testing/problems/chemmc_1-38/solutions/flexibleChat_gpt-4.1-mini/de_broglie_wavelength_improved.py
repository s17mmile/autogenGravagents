# filename: de_broglie_wavelength_improved.py

from scipy.constants import h, m_e, e

def calculate_de_broglie_wavelength(kinetic_energy_ev):
    """
    Calculate the de Broglie wavelength of an electron given its kinetic energy in electronvolts.

    Parameters:
    kinetic_energy_ev (float): Kinetic energy in electronvolts (must be positive).

    Returns:
    float: de Broglie wavelength in meters.
    """
    if kinetic_energy_ev <= 0:
        raise ValueError("Kinetic energy must be positive.")

    # Convert kinetic energy from eV to Joules
    kinetic_energy_joule = kinetic_energy_ev * e

    # Calculate momentum p = sqrt(2*m*K)
    p = (2 * m_e * kinetic_energy_joule) ** 0.5

    # Calculate de Broglie wavelength lambda = h / p
    wavelength = h / p

    return wavelength

# Given kinetic energy
kinetic_energy_ev = 100

# Calculate wavelength
wavelength = calculate_de_broglie_wavelength(kinetic_energy_ev)

# Output the result
print(f"De Broglie wavelength for an electron with {kinetic_energy_ev} eV kinetic energy: {wavelength:.3e} meters")
