# filename: probability_position_measurement.py
import numpy as np

def probability_between(x, c, interval_width):
    """Calculate the probability that the particle's position lies between x and x + interval_width."""
    prefactor = np.sqrt(32 / (np.pi * c**6))
    x_squared = x**2
    exponential = np.exp(-2 * x_squared / c**2)
    probability_density = prefactor * x_squared * exponential
    return probability_density * interval_width

# Constants
c = 2.000  # Angstroms
x = 2.000  # Angstroms
interval_width = 0.001  # Angstroms

probability = probability_between(x, c, interval_width)
print(f"Probability that the position lies between {x} Å and {x + interval_width} Å: {probability:.6e}")
