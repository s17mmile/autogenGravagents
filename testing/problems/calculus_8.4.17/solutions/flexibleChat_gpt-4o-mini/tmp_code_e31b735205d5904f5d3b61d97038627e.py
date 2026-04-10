import numpy as np
from scipy.integrate import quad

# Constants
D = 6  # total amount of dye injected in mg


def concentration(t):
    """Calculate the dye concentration at time t.
    
    Args:
        t (float): Time in seconds.
    
    Returns:
        float: Concentration in mg/L.
    """
    return 20 * t * np.exp(-0.6 * t)

# Time interval
T = 10  # seconds

# Calculate the integral of concentration from 0 to T
integral_result, _ = quad(concentration, 0, T)

# Check for zero integral result
if integral_result == 0:
    raise ValueError('Integral result is zero, cannot compute cardiac output.')

# Calculate cardiac output
cardiac_output = D / integral_result

# Output the result with formatting
print(f'Cardiac Output: {cardiac_output:.2f} mL/min')