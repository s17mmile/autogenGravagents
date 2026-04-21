# filename: barometric_pressure_calculation.py
import math

def barometric_pressure(p0, h, T, M=0.029, g=9.81, R=8.314):
    """
    Calculate atmospheric pressure at altitude h using the barometric formula.

    Parameters:
    p0 : float - pressure at sea level in Pascals
    h : float - altitude in meters
    T : float - absolute temperature in Kelvin
    M : float - molar mass of air in kg/mol (default 0.029)
    g : float - acceleration due to gravity in m/s^2 (default 9.81)
    R : float - universal gas constant in J/(mol*K) (default 8.314)

    Returns:
    p : float - pressure at altitude h in Pascals
    """
    exponent = -(M * g * h) / (R * T)
    p = p0 * math.exp(exponent)
    return p

# Constants
p0 = 101325  # sea level pressure in Pascals (1 atm)
h = 11000    # altitude in meters (11 km)
T = 220      # temperature in Kelvin at 11 km altitude

pressure_at_11km = barometric_pressure(p0, h, T)
print(f"Atmospheric pressure at {h} m altitude: {pressure_at_11km:.2f} Pa (~{pressure_at_11km / 101325:.2f} atm)")
