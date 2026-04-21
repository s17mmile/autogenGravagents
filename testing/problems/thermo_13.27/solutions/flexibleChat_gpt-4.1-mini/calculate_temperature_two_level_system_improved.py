# filename: calculate_temperature_two_level_system_improved.py
import math
from scipy.constants import Boltzmann

def calculate_temperature(delta_E, population_ratio):
    """Calculate the temperature at which the population ratio of ground to excited state is given.

    Args:
        delta_E (float): Energy difference between states in Joules (must be positive).
        population_ratio (float): Ratio N_ground / N_excited (must be > 1).

    Returns:
        float: Temperature in Kelvin.

    Raises:
        ValueError: If inputs are not positive or population_ratio <= 1.
    """
    if delta_E <= 0:
        raise ValueError("Energy difference delta_E must be positive.")
    if population_ratio <= 1:
        raise ValueError("Population ratio must be greater than 1.")

    T = delta_E / (Boltzmann * math.log(population_ratio))
    return T

# Given values
energy_difference = 1.30e-18  # in Joules
population_ratio = 5  # N_ground / N_excited

# Calculate temperature
temperature = calculate_temperature(energy_difference, population_ratio)

# Print the result
print(f"Temperature at which ground state population is 5 times excited state: {temperature:.2f} K")
