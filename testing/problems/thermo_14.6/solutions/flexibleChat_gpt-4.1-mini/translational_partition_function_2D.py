# filename: translational_partition_function_2D.py

import math

def translational_partition_function_2D(T=298, area_cm2=1.00):
    """Calculate the 2D translational partition function for argon gas.

    Parameters:
    T (float): Temperature in kelvin (must be positive).
    area_cm2 (float): Area in square centimeters (must be positive).

    Returns:
    float: The translational partition function (dimensionless).
    """
    if T <= 0:
        raise ValueError("Temperature must be positive.")
    if area_cm2 <= 0:
        raise ValueError("Area must be positive.")

    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    k_B = 1.381e-23  # Boltzmann constant in J/K
    atomic_mass_unit = 1.6605e-27  # kg
    m_Ar = 39.95 * atomic_mass_unit  # mass of argon atom in kg

    # Convert area from cm^2 to m^2
    A = area_cm2 * 1e-4

    # Calculate thermal de Broglie wavelength
    Lambda = h / math.sqrt(2 * math.pi * m_Ar * k_B * T)

    # Calculate translational partition function in 2D
    q_trans = A / (Lambda ** 2)

    return q_trans

# Calculate and print the translational partition function
q_trans_value = translational_partition_function_2D()
print(f'Translational partition function (2D) for Ar at 298 K and 1.00 cm^2: {q_trans_value:.3e}')
