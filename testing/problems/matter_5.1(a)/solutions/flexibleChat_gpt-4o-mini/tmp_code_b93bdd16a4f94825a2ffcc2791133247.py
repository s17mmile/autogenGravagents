import numpy as np
from scipy.integrate import quad

# Constants
L = 10.0  # Length of the nanotube in nm
x1 = 4.95  # Lower limit of the interval in nm
x2 = 5.05  # Upper limit of the interval in nm

# Wavefunction squared function
def calculate_wavefunction_squared(x):
    """Calculate the square of the wavefunction at position x."""
    return (2 / L) * (np.sin(np.pi * x / L) ** 2)

# Calculate the probability
try:
    probability, _ = quad(calculate_wavefunction_squared, x1, x2)
    # Print the result as a percentage
    print(f'Probability of finding the electron between {x1} nm and {x2} nm: {probability * 100:.2f}%')
except Exception as e:
    print(f'An error occurred during integration: {e}')