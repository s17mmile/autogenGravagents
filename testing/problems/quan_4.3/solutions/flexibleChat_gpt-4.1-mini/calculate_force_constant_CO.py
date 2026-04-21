# filename: calculate_force_constant_CO.py

import math

def calculate_force_constant(wavenumber_cm=2143, m_C=12.0, m_O=16.0):
    """
    Calculate the force constant k of a diatomic molecule given the infrared band wavenumber.

    Parameters:
    wavenumber_cm (float): Wavenumber of the infrared band in cm^-1.
    m_C (float): Atomic mass of carbon in atomic mass units (amu).
    m_O (float): Atomic mass of oxygen in atomic mass units (amu).

    Returns:
    float: Force constant k in N/m.
    """
    c = 2.998e10  # Speed of light in cm/s
    amu_to_kg = 1.6605e-27  # Atomic mass unit to kg

    # Calculate reduced mass in kg
    mu = (m_C * m_O) / (m_C + m_O) * amu_to_kg

    # Calculate force constant k using the formula
    k = (2 * math.pi * c * wavenumber_cm)**2 * mu

    return k

if __name__ == "__main__":
    force_constant = calculate_force_constant()
    print(f"Force constant of 12C16O: {force_constant:.2f} N/m")
