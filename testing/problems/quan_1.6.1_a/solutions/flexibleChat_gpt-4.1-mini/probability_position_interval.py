# filename: probability_position_interval.py
import numpy as np

def probability_in_interval(a, x, delta_x):
    """Calculate the probability that the particle's position lies between x and x + delta_x.
    Assumes wavefunction Psi = a^{-1/2} exp(-|x|/a) with a in nm and x in nm."""
    # Probability density function P(x) = (1/a) * exp(-2|x|/a)
    P_x = (1 / a) * np.exp(-2 * np.abs(x) / a)
    return P_x * delta_x

# Constants in nm
a = 1.0
x = 1.5
delta_x = 0.0001

probability = probability_in_interval(a, x, delta_x)
print(f"Probability that the particle is between {x} nm and {x + delta_x} nm: {probability:.6e}")
