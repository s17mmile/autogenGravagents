# filename: hydrogen_transition_properties.py

from scipy.constants import Rydberg, h, c, electron_volt

def hydrogen_transition_properties(n1, n2):
    """Calculate wavenumber (cm^-1), wavelength (m), and photon energy (eV) for hydrogen transition n2->n1.

    Args:
        n1 (int): Lower energy level (must be positive integer).
        n2 (int): Higher energy level (must be positive integer and greater than n1).

    Returns:
        tuple: (wavenumber_cm_inv, wavelength_m, energy_eV)
    """
    if not (isinstance(n1, int) and isinstance(n2, int)):
        raise TypeError("n1 and n2 must be integers.")
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive integers.")
    if n2 <= n1:
        raise ValueError("n2 must be greater than n1 for emission transition.")

    # Calculate wavenumber in m^-1 using scipy.constants.Rydberg
    wavenumber_m_inv = Rydberg * (1 / n1**2 - 1 / n2**2)

    # Convert wavenumber to cm^-1
    wavenumber_cm_inv = wavenumber_m_inv / 100

    # Calculate wavelength in meters
    wavelength_m = 1 / wavenumber_m_inv

    # Calculate photon energy in Joules
    energy_J = h * c / wavelength_m

    # Convert energy to electronvolts
    energy_eV = energy_J / electron_volt

    return wavenumber_cm_inv, wavelength_m, energy_eV


if __name__ == '__main__':
    n1 = 1
    n2 = 2
    wavenumber, wavelength, energy = hydrogen_transition_properties(n1, n2)
    print(f'Wavenumber (cm^-1): {wavenumber:.2f}')
    print(f'Wavelength (nm): {wavelength * 1e9:.2f}')
    print(f'Photon Energy (eV): {energy:.2f}')
