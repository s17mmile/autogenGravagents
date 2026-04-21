# filename: vibrational_partition_function_temperature.py
import numpy as np
from scipy.constants import h, c, k
from scipy.optimize import root_scalar

# Vibrational wavenumber of Br2 in cm^-1
wavenumber_cm = 323.2

# Convert wavenumber to frequency in Hz (1 cm^-1 = c * 100 Hz)
frequency = wavenumber_cm * c * 100

def q_vib_exact(T):
    """Calculate the exact vibrational partition function at temperature T (K)."""
    if T <= 0:
        raise ValueError("Temperature must be positive.")
    beta = h * frequency / (k * T)
    return 1 / (1 - np.exp(-beta))

def q_vib_approx(T):
    """Calculate the approximate vibrational partition function at temperature T (K)."""
    if T <= 0:
        raise ValueError("Temperature must be positive.")
    return k * T / (h * frequency)

def relative_difference(T):
    """Calculate the relative difference between exact and approximate partition functions."""
    q_exact = q_vib_exact(T)
    q_approx = q_vib_approx(T)
    return abs(q_exact - q_approx) / q_exact

# Function to find root for: relative_difference(T) - 0.05 = 0
# Search temperature range from 1 K to 2000 K

def func(T):
    return relative_difference(T) - 0.05

solution = root_scalar(func, bracket=[1, 2000], method='bisect')

if solution.converged:
    temperature_5_percent = solution.root
    print(f"Temperature at which the vibrational partition function's exact and approximate values differ by 5%: {temperature_5_percent:.2f} K")
else:
    raise RuntimeError("No solution found in the given temperature range.")
