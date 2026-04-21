# filename: calculate_co2_cross_section.py
import math
from scipy.constants import Boltzmann, Avogadro, pi

def calculate_collisional_cross_section(D, T, p, Omega_D=1):
    """
    Estimate the effective molecular diameter (collisional cross section diameter) of CO2
    from the diffusion coefficient using the Chapman-Enskog equation approximation.

    Parameters:
    D : float
        Diffusion coefficient in m^2/s
    T : float
        Temperature in Kelvin
    p : float
        Pressure in Pascals
    Omega_D : float, optional
        Collision integral (dimensionless), default is 1 for estimation

    Returns:
    sigma : float
        Effective molecular diameter (collisional cross section diameter) in meters
    """
    if D <= 0 or T <= 0 or p <= 0:
        raise ValueError("Diffusion coefficient, temperature, and pressure must be positive values.")

    # Molar mass of CO2 in kg/mol
    M_CO2 = 44.01e-3

    # Molecular mass per molecule in kg
    m_molecule = M_CO2 / Avogadro

    numerator = (3/16) * (Boltzmann * T)**1.5
    denominator = p * D * math.sqrt(1/(2 * pi * m_molecule)) * Omega_D

    sigma_squared = numerator / denominator

    sigma = math.sqrt(sigma_squared)

    return sigma

# Given values
D = 1.00e-5  # m^2/s
T = 273       # K
p = 1.013e5   # Pa (1 atm)

sigma = calculate_collisional_cross_section(D, T, p)

print(f'Estimated collisional cross section diameter of CO2: {sigma:.3e} m')
