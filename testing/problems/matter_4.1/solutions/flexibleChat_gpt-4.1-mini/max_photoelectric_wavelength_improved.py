# filename: max_photoelectric_wavelength_improved.py

from scipy.constants import h, c, electron_volt

def calculate_max_wavelength(incident_wavelength_nm, kinetic_energy_eV):
    """
    Calculate the maximum wavelength (in nm) capable of ejecting an electron from a metal
    given the incident photon wavelength and the kinetic energy of the ejected electron.

    Parameters:
    incident_wavelength_nm (float): Wavelength of incident photon in nanometers.
    kinetic_energy_eV (float): Kinetic energy of the ejected electron in electron volts.

    Returns:
    float: Maximum wavelength in nanometers capable of ejecting an electron.

    Raises:
    ValueError: If kinetic energy is greater than or equal to the photon energy.
    """
    # Convert incident wavelength from nm to meters
    incident_wavelength_m = incident_wavelength_nm * 1e-9

    # Calculate energy of incident photon in Joules
    E_photon = h * c / incident_wavelength_m

    # Convert kinetic energy from eV to Joules
    KE = kinetic_energy_eV * electron_volt

    # Validate energies
    if KE >= E_photon:
        raise ValueError("Kinetic energy cannot be greater than or equal to the photon energy.")

    # Calculate work function (phi) in Joules
    phi = E_photon - KE

    # Calculate maximum wavelength (lambda_max) in meters
    lambda_max_m = h * c / phi

    # Convert maximum wavelength back to nm
    lambda_max_nm = lambda_max_m * 1e9

    return lambda_max_nm

# Given values
incident_wavelength_nm = 305
kinetic_energy_eV = 1.77

# Calculate maximum wavelength
max_wavelength = calculate_max_wavelength(incident_wavelength_nm, kinetic_energy_eV)

# Print the result
print(f"Maximum wavelength capable of ejecting an electron: {max_wavelength:.2f} nm")
