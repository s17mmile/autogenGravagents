import numpy as np
from numpy import exp, pi
from scipy.integrate import quad

# Constants
c = 2.000  # Å
position_start = 2.000  # Å
position_end = 2.001  # Å

# Probability density function
def P(x):
    """
    Calculate the probability density at position x.
    
    Parameters:
    x (float): The position at which to evaluate the probability density.
    
    Returns:
    float: The probability density at position x.
    """
    coeff = (32 / (pi * c**6))**0.5
    return coeff * x**2 * exp(-2 * x**2 / c**2)

# Check for valid input
if position_start >= position_end:
    raise ValueError('position_start must be less than position_end')

# Estimate the probability between position_start and position_end
probability, _ = quad(P, position_start, position_end)

# Output the estimated probability
print(f'Estimated probability of finding the particle between {position_start:.3f} Å and {position_end:.3f} Å: {probability:.6f}')