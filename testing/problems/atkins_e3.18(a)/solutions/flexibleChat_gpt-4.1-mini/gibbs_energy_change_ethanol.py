# filename: gibbs_energy_change_ethanol.py

import math

# Constants
R = 8.314  # J/(mol*K), ideal gas constant
T = 298.15  # K, temperature
P1 = 1  # atm, initial pressure
P2 = 3000  # atm, final pressure

# Given data
mass_ethanol = 35.0  # g
molar_mass_ethanol = 46.07  # g/mol
# Density is given but not used in this calculation as mass and molar mass suffice
# density_ethanol = 0.789  # g/cm^3

def calculate_delta_g(mass, molar_mass, T, P1, P2, R):
    """
    Calculate the change in Gibbs energy for an ideal gas during an isothermal pressure change.

    Parameters:
    mass (float): mass of the substance in grams
    molar_mass (float): molar mass of the substance in g/mol
    T (float): temperature in Kelvin
    P1 (float): initial pressure in atm
    P2 (float): final pressure in atm
    R (float): ideal gas constant in J/(mol*K)

    Returns:
    float: change in Gibbs energy in Joules
    """
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Pressures must be positive values.")
    if P2 <= P1:
        raise ValueError("Final pressure must be greater than initial pressure for this calculation.")

    n = mass / molar_mass  # moles
    delta_g = n * R * T * math.log(P2 / P1)
    return delta_g

# Calculate delta G
delta_g = calculate_delta_g(mass_ethanol, molar_mass_ethanol, T, P1, P2, R)

# Output the result
print(f"Change in Gibbs energy (Delta G) for {mass_ethanol} g ethanol when pressure increases from {P1} atm to {P2} atm at {T} K: {delta_g:.2f} J")
