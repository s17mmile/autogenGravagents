import numpy as np
from scipy.integrate import quad

# Define the parameter a
# a is the scale parameter for the wave function
# Here, it is set to 1.0000 nm

a = 1.0000  # nm

# Define the probability density function
# This function calculates the probability density based on the wave function

def probability_density(x):
    """Returns the probability density for a given x."""
    return np.exp(-2 * x)  # since x is positive in the range of interest

# Define the limits of integration
x_lower = 1.5000  # nm
x_upper = 1.5001  # nm

# Calculate the probability by integrating the probability density over the interval
try:
    probability, _ = quad(probability_density, x_lower, x_upper)
    # Print the result
    print(f'Probability that the measured value lies between {x_lower} nm and {x_upper} nm: {probability} (dimensionless)')
except Exception as e:
    print(f'An error occurred during integration: {e}')