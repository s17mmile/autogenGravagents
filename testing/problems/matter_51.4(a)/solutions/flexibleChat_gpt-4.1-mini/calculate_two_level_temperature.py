# filename: calculate_two_level_temperature.py
from math import log
from scipy.constants import h, c, k

def calculate_temperature(wavenumber_cm_inv, population_ratio):
    """
    Calculate the temperature of a two-level system given the energy separation in wavenumbers
    and the population ratio of the upper to lower state.

    Parameters:
    wavenumber_cm_inv (float): Energy separation in cm^-1
    population_ratio (float): Ratio of upper state population to lower state population (0 < ratio < 1)

    Returns:
    float: Temperature in Kelvin

    Raises:
    ValueError: If population_ratio is not between 0 and 1 (exclusive)
    """
    if not (0 < population_ratio < 1):
        raise ValueError("population_ratio must be between 0 and 1 (exclusive)")

    # Convert wavenumber from cm^-1 to m^-1
    wavenumber_m_inv = wavenumber_cm_inv * 100
    # Calculate energy difference in Joules
    delta_E = h * c * wavenumber_m_inv
    # Calculate temperature using Boltzmann distribution
    temperature = delta_E / (k * (-log(population_ratio)))
    return temperature

# Given values
wavenumber = 400  # in cm^-1
population_ratio = 1/3  # N_upper / N_lower

# Calculate temperature
temperature_K = calculate_temperature(wavenumber, population_ratio)

# Print the result
print(f"Temperature of the two-level system: {temperature_K:.2f} K")
