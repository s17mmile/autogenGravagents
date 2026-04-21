# filename: barometric_pressure_difference.py
import math

def barometric_pressure_difference(p0: float, h: float, M: float, g: float, R: float, T: float) -> tuple[float, float]:
    """Calculate pressure at height h using the barometric formula and return pressure difference.

    Parameters:
    p0 (float): Pressure at sea level in Pascals (Pa)
    h (float): Height above sea level in meters (m)
    M (float): Molar mass of the gas in kg/mol
    g (float): Acceleration due to gravity in m/s^2
    R (float): Universal gas constant in J/(mol*K)
    T (float): Absolute temperature in Kelvin (K)

    Returns:
    tuple: (pressure at height h in Pa, pressure difference p0 - p in Pa)
    """
    if h < 0:
        raise ValueError("Height h must be non-negative")
    exponent = -(M * g * h) / (R * T)
    p = p0 * math.exp(exponent)
    delta_p = p0 - p
    return p, delta_p

# Constants
p0 = 101325  # Sea level standard atmospheric pressure in Pa
h = 0.15     # Height in meters (15 cm)
M = 0.029    # Molar mass of air in kg/mol
T = 298      # Temperature in Kelvin (25 C)
g = 9.81     # Acceleration due to gravity in m/s^2
R = 8.314    # Universal gas constant in J/(mol*K)

pressure_top, pressure_difference = barometric_pressure_difference(p0, h, M, g, R, T)

print(f"Pressure at top of vessel: {pressure_top:.2f} Pa")
print(f"Pressure difference (bottom - top): {pressure_difference:.2f} Pa")
