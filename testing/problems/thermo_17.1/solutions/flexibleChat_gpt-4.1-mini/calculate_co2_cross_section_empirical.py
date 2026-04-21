# filename: calculate_co2_cross_section_empirical.py
import math

def calculate_collisional_cross_section(D, T, p, M, Omega_D=1):
    """
    Estimate the collisional cross section diameter (sigma) of CO2 from the diffusion coefficient
    using an empirical formula with consistent units.

    Parameters:
    D : float
        Diffusion coefficient in m^2/s
    T : float
        Temperature in Kelvin
    p : float
        Pressure in atm
    M : float
        Molar mass in g/mol
    Omega_D : float, optional
        Collision integral (dimensionless), default is 1

    Returns:
    sigma_m : float
        Collisional cross section diameter in meters
    """
    if D <= 0 or T <= 0 or p <= 0 or M <= 0:
        raise ValueError("All input parameters must be positive.")

    # Empirical constant in m^2 K^{3/2} / s
    constant = 1.858e-7

    # Calculate sigma in angstroms
    sigma_angstrom = math.sqrt((constant * T**1.5) / (p * D * Omega_D * math.sqrt(M)))

    # Convert angstroms to meters (1 angstrom = 1e-10 m)
    sigma_m = sigma_angstrom * 1e-10

    return sigma_m

# Given values
D = 1.00e-5  # m^2/s
T = 273       # K
p = 1.0       # atm
M = 44.01     # g/mol (CO2 molar mass)

sigma = calculate_collisional_cross_section(D, T, p, M)

print(f'Estimated collisional cross section diameter of CO2: {sigma:.3e} m')
