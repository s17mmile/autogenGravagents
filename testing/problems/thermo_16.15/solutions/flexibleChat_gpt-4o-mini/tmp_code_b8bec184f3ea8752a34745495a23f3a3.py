import numpy as np
from scipy.integrate import quad

# Constants
k = 1.38e-23  # Boltzmann constant in J/K
T = 298  # Temperature in K
molar_mass_N2 = 28e-3  # Molar mass of N2 in kg/mol
N_A = 6.022e23  # Avogadro's number in molecules/mol
m = molar_mass_N2 / N_A  # Mass of a single N2 molecule in kg

# Maxwell-Boltzmann distribution function

def f(v):
    """Probability density function for the speed of nitrogen molecules.
    Args:
        v (float): Speed of the molecule.
    Returns:
        float: Probability density at speed v.
    """
    return (m / (2 * np.pi * k * T))**(3/2) * 4 * np.pi * v**2 * np.exp(-m * v**2 / (2 * k * T))

def calculate_fraction(v1, v2):
    """Calculate the fraction of molecules with speeds between v1 and v2.
    Args:
        v1 (float): Lower speed limit in m/s.
        v2 (float): Upper speed limit in m/s.
    Returns:
        float: Fraction of molecules with speeds between v1 and v2.
    """
    try:
        P_less_than_v2, _ = quad(f, 0, v2)
        P_less_than_v1, _ = quad(f, 0, v1)
        return P_less_than_v2 - P_less_than_v1
    except Exception as e:
        print(f"Error during integration: {e}")
        return None

# Main function to execute the calculation
if __name__ == '__main__':
    v1 = 200  # lower speed limit in m/s
    v2 = 300  # upper speed limit in m/s
    fraction = calculate_fraction(v1, v2)
    if fraction is not None:
        print(f"Fraction of nitrogen molecules with speeds between {v1} m/s and {v2} m/s: {fraction}")