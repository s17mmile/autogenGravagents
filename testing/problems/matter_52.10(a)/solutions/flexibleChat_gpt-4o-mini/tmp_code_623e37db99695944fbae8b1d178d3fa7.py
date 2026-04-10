import numpy as np

# Constants
h = 6.626e-34  # Planck's constant in J·s
k = 1.38e-23   # Boltzmann's constant in J/K
c = 3.00e10    # Speed of light in cm/s
wavenumber = 323.2  # in cm^-1

# Convert wavenumber to frequency
frequency = wavenumber * c  # in Hz

# Function to calculate exact and approximate partition functions

def partition_functions(T):
    """
    Calculate the exact and approximate vibrational partition functions.
    
    Parameters:
    T : float
        Temperature in Kelvin.
    
    Returns:
    tuple : (q_exact, q_approx)
        Exact and approximate partition functions.
    """
    if T <= 0:
        raise ValueError('Temperature must be greater than 0 K')
    beta = h * frequency / (k * T)
    q_exact = 1 / (1 - np.exp(-beta))
    q_approx = (k * T) / (h * frequency)
    return q_exact, q_approx

# Find temperature where the difference is within 5%
results = []
for T in range(1, 1000):  # Check from 1K to 1000K
    q_exact, q_approx = partition_functions(T)
    if abs(q_exact - q_approx) < 0.05 * q_approx:
        results.append((T, q_exact, q_approx))

# Print results
for result in results:
    print(f'Temperature: {result[0]} K, Exact q: {result[1]}, Approx q: {result[2]}')