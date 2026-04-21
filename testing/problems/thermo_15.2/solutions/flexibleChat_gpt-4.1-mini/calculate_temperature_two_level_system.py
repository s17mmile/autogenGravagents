# filename: calculate_temperature_two_level_system.py
import math
from scipy.constants import h, k

# Frequency of the two-level system in s^-1
v = 1.50e13

def calculate_temperature(frequency):
    """
    Calculate the temperature T at which the internal energy U = 0.25 N hv
    for a two-level system with frequency 'frequency'.

    Formula derived from:
    U = N hv / (1 + exp(hv/kT)) = 0.25 N hv
    => exp(hv/kT) = 3
    => T = hv / (k * ln(3))
    """
    return (h * frequency) / (k * math.log(3))

# Calculate temperature
T = calculate_temperature(v)

print(f"Temperature at which internal energy is 0.25 N hv: {T:.2f} K")
