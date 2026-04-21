# filename: calculate_debye_length_improved.py
import math
from scipy.constants import epsilon_0, Boltzmann, Avogadro, elementary_charge

def calculate_debye_length(T=298, c_K3PO4=0.0075, epsilon_r=78.5):
    """
    Calculate the Debye-Hückel screening length (1/kappa) in meters for a given temperature,
    concentration of K3PO4 solution (mol/L), and relative permittivity.

    Parameters:
    - T: Temperature in Kelvin (default 298 K)
    - c_K3PO4: Concentration of K3PO4 in mol/L (default 0.0075 M)
    - epsilon_r: Relative permittivity of the solvent (default 78.5 for water at 298 K)

    Returns:
    - Debye length in meters
    """
    # Ion charges
    z_K = 1
    z_PO4 = -3

    # Ion concentrations in mol/L
    c_K = 3 * c_K3PO4
    c_PO4 = c_K3PO4

    # Calculate ionic strength I = 0.5 * sum(c_i * z_i^2)
    I = 0.5 * (c_K * z_K**2 + c_PO4 * z_PO4**2)  # mol/L

    # Assert ionic strength is positive and reasonable
    assert I > 0, "Ionic strength must be positive"

    # Convert ionic strength to mol/m^3
    I_m3 = I * 1000  # mol/m^3

    numerator = epsilon_0 * epsilon_r * Boltzmann * T
    denominator = 2 * Avogadro * elementary_charge**2 * I_m3

    debye_length = math.sqrt(numerator / denominator)  # meters

    return debye_length

if __name__ == '__main__':
    length = calculate_debye_length()
    print(f'Debye-Hückel screening length (1/kappa) at 298 K: {length:.3e} meters')
