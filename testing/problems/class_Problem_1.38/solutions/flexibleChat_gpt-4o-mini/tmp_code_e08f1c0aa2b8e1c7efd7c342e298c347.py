import numpy as np
from scipy.integrate import quad

# Define the function to integrate

def integrand_sin_squared(t):
    """Returns the value of -sin^2(t) for integration."""
    return -np.sin(t)**2

try:
    # Compute the integral over the interval [0, 2*pi]
    result, _ = quad(integrand_sin_squared, 0, 2 * np.pi)
    # Output the result with context
    print('The value of the integral \( \int_C \mathbf{A} \cdot d\mathbf{r} \) is:', result)
except Exception as e:
    print('An error occurred during integration:', e)