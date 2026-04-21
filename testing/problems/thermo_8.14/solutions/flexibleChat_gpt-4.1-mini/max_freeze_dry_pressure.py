# filename: max_freeze_dry_pressure.py

import math

# Define gas constant as a global constant
R = 8.314  # J/(mol*K)

def calculate_max_pressure(T1, P1, T2, delta_H_sub, R=R):
    """
    Calculate vapor pressure P2 at temperature T2 using Clausius-Clapeyron equation.

    Parameters:
    - T1: Reference temperature in Kelvin (must be > 0)
    - P1: Vapor pressure at T1 in Pascals (must be > 0)
    - T2: Target temperature in Kelvin (must be > 0)
    - delta_H_sub: Enthalpy of sublimation in J/mol (must be > 0)
    - R: Gas constant in J/(mol*K)

    Returns:
    - P2: Vapor pressure at T2 in Pascals

    Note: Assumes delta_H_sub is constant over the temperature range.
    """
    # Input validation
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperatures must be greater than zero Kelvin.")
    if P1 <= 0:
        raise ValueError("Pressure P1 must be greater than zero.")
    if delta_H_sub <= 0:
        raise ValueError("Enthalpy of sublimation must be greater than zero.")

    ln_P2_P1 = -delta_H_sub / R * (1/T2 - 1/T1)
    P2 = P1 * math.exp(ln_P2_P1)
    return P2

# Given data
T1 = 273.16  # K
P1 = 624     # Pa
T2 = 268.15  # K (-5.00 C)
delta_H_sub = 51000  # J/mol (enthalpy of sublimation of ice)

max_pressure = calculate_max_pressure(T1, P1, T2, delta_H_sub)

print(f"Maximum pressure for freeze drying at -5.00 C: {max_pressure:.2f} Pa")
