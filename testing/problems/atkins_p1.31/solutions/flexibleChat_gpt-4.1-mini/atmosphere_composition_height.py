# filename: atmosphere_composition_height.py
import math

def mole_fraction_from_mass_fraction(mass_frac_N2, M_N2, M_O2):
    """Calculate mole fractions of N2 and O2 from given mass fraction of N2."""
    numerator = mass_frac_N2 * M_O2
    denominator = M_N2 - mass_frac_N2 * M_N2 + mass_frac_N2 * M_O2
    x_N2 = numerator / denominator
    x_O2 = 1 - x_N2
    return x_N2, x_O2

def calculate_height_and_pressure():
    """
    Calculate the height above Earth's surface where the atmospheric composition
    is 90% nitrogen and 10% oxygen by mass, assuming constant temperature at 25 C.
    Returns:
        h (float): Height in meters
        P_total_h (float): Atmospheric pressure at height h in Pascals
    """
    # Constants
    T = 298.15  # Temperature in K (25 degrees Celsius)
    g = 9.81  # gravitational acceleration in m/s^2
    R = 8.314  # universal gas constant J/(mol K)

    # Molecular masses in kg/mol
    M_N2 = 28.0e-3  # Nitrogen
    M_O2 = 32.0e-3  # Oxygen

    # Surface composition by mass
    mass_frac_N2_surface = 0.80
    mass_frac_O2_surface = 0.20

    # Validate input mass fractions
    if not (0 <= mass_frac_N2_surface <= 1 and 0 <= mass_frac_O2_surface <= 1):
        raise ValueError('Surface mass fractions must be between 0 and 1.')
    if abs(mass_frac_N2_surface + mass_frac_O2_surface - 1) > 1e-6:
        raise ValueError('Surface mass fractions must sum to 1.')

    # Desired composition by mass at height h
    mass_frac_N2_target = 0.90
    mass_frac_O2_target = 0.10

    if not (0 <= mass_frac_N2_target <= 1 and 0 <= mass_frac_O2_target <= 1):
        raise ValueError('Target mass fractions must be between 0 and 1.')
    if abs(mass_frac_N2_target + mass_frac_O2_target - 1) > 1e-6:
        raise ValueError('Target mass fractions must sum to 1.')

    # Calculate scale heights for N2 and O2 (meters)
    H_N2 = R * T / (M_N2 * g)
    H_O2 = R * T / (M_O2 * g)

    # Calculate mole fractions at surface from mass fractions
    x_N2_surface, x_O2_surface = mole_fraction_from_mass_fraction(mass_frac_N2_surface, M_N2, M_O2)

    # Partial pressures at surface proportional to mole fractions
    P0 = 101325  # Standard atmospheric pressure in Pascals
    P_N2_0 = x_N2_surface * P0
    P_O2_0 = x_O2_surface * P0

    # Solve for height h where nitrogen mass fraction is 90%
    ratio = (mass_frac_N2_target * P_O2_0 * M_O2) / (mass_frac_O2_target * P_N2_0 * M_N2)
    exponent = math.log(ratio)
    h = exponent / (1/H_O2 - 1/H_N2)

    # Calculate total pressure at height h
    P_N2_h = P_N2_0 * math.exp(-h / H_N2)
    P_O2_h = P_O2_0 * math.exp(-h / H_O2)
    P_total_h = P_N2_h + P_O2_h

    return h, P_total_h

if __name__ == '__main__':
    height, pressure = calculate_height_and_pressure()
    print(f'Height where atmosphere is 90% N2 by mass: {height:.2f} meters')
    print(f'Pressure at that height: {pressure:.2f} Pascals')
