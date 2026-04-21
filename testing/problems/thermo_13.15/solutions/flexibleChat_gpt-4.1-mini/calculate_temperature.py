# filename: calculate_temperature.py
from math import log
from scipy.constants import h, c, k

# Given values
population_ratio = 8  # N0/N1
wavenumber_cm = 6000  # in cm^-1

# Convert wavenumber from cm^-1 to m^-1 by multiplying by 100
wavenumber_m = wavenumber_cm * 100

# Calculate energy difference in Joules
delta_E = h * c * wavenumber_m

# Calculate temperature using Boltzmann distribution
T = delta_E / (k * log(population_ratio))

print(f"Temperature of the collection: {T:.2f} K")
