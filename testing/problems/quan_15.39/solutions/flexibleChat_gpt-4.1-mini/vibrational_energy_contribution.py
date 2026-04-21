# filename: vibrational_energy_contribution.py
import math
from scipy.constants import h, c, k, R

def vibrational_energy_contribution(wavenumber_cm_inv, temperature_K=298.15):
    """
    Calculate the vibrational contribution to the molar internal energy (excluding zero-point energy) 
    for a single vibrational mode of a nonlinear molecule.

    Parameters:
    wavenumber_cm_inv (float): Vibrational wavenumber in cm^-1 (must be positive).
    temperature_K (float): Temperature in Kelvin (must be positive). Default is 298.15 K (25 C).

    Returns:
    float: Vibrational contribution to molar internal energy in J/mol.
    """
    if wavenumber_cm_inv <= 0:
        raise ValueError("Wavenumber must be positive.")
    if temperature_K <= 0:
        raise ValueError("Temperature must be positive.")

    # Convert speed of light from m/s to cm/s for consistency with wavenumber
    c_cm_s = c * 100  # c is in m/s, convert to cm/s

    # Calculate frequency nu_s in Hz
    nu_s = wavenumber_cm_inv * c_cm_s

    # Calculate characteristic temperature theta_s
    theta_s = (h * nu_s) / k

    # Use expm1 for numerical stability: exp(x) - 1
    denominator = math.expm1(theta_s / temperature_K)
    if denominator == 0:
        # Avoid division by zero, return 0 contribution
        return 0.0

    # Calculate vibrational contribution to molar internal energy
    U_m_vib = R * (theta_s / denominator)

    return U_m_vib

# Given wavenumber 900 cm^-1 and temperature 25 C (298.15 K)
wavenumber = 900
temperature = 298.15

U_m_vib_value = vibrational_energy_contribution(wavenumber, temperature)

print(f"Vibrational contribution to molar internal energy at {temperature} K for wavenumber {wavenumber} cm^-1: {U_m_vib_value:.4f} J/mol")
