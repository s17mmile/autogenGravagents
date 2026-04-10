import math

# Constants
plancks_constant = 6.626e-34  # Planck's constant in Js
boltzmann_constant = 1.38e-23  # Boltzmann's constant in J/K
avogadro_number = 6.022e23  # Avogadro's number in mol^-1
volume = 1e-3  # Volume in m^3 (1000 cm^3)

# Mass of one molecule of O2 (molar mass = 32 g/mol)
m = 32e-3 / avogadro_number  # mass in kg

# Calculate temperature T
T = (plancks_constant**2 / (2 * math.pi * m * boltzmann_constant)) * ((avogadro_number * plancks_constant**3 / volume)**(2/3))

# Output the result
print(f'Temperature: {T} K')