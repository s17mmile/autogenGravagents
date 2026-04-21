# filename: calculate_bond_distance_CS.py
import math

def calculate_bond_length(frequency_MHz, mass1_u, mass2_u):
    """
    Calculate bond length from rotational absorption frequency and atomic masses.
    :param frequency_MHz: Rotational absorption frequency in MHz
    :param mass1_u: Mass of first atom in atomic mass units (u)
    :param mass2_u: Mass of second atom in atomic mass units (u)
    :return: Bond length in meters
    """
    # Convert frequency from MHz to Hz
    frequency_Hz = frequency_MHz * 1e6

    # Constants
    h = 6.62607015e-34  # Planck's constant in J*s
    amu_kg = 1.66053906660e-27  # atomic mass unit in kg

    # Calculate rotational constant B in Hz
    B = frequency_Hz / 2

    # Calculate reduced mass in kg
    mu = (mass1_u * mass2_u) / (mass1_u + mass2_u) * amu_kg

    # Calculate moment of inertia I in kg*m^2
    I = h / (8 * math.pi**2 * B)

    # Calculate bond length r in meters
    r = math.sqrt(I / mu)

    return r

# Given data for 12C32S
frequency_MHz = 48991.0
mass_C = 12.0
mass_S = 32.0

bond_length = calculate_bond_length(frequency_MHz, mass_C, mass_S)
print(f"Bond length in 12C32S molecule: {bond_length:.3e} meters")
