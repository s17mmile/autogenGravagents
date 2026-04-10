import math

# Constants
E = 1.30e-18  # Energy separation in Joules
k = 1.38e-23  # Boltzmann constant in J/K

# Calculate temperature
ln_5_value = math.log(5)
T = E / (k * ln_5_value)

# Print the result with context
print(f'Calculated temperature at which the ground state population is 5 times that of the excited state: {T:.2f} K')