# filename: calculate_energy_bars.py

import math

# Constants with units
mass_kg = 60  # mass of hiker in kilograms
height_m = 828  # height of Burj Khalifa in meters
g = 9.8  # acceleration due to gravity in meters per second squared
efficiency = 0.25  # body efficiency (fraction)
energy_per_bar_kJ = 1.08e3  # energy per bar in kilojoules

# Calculate work done (potential energy gained) in Joules
work_joules = mass_kg * g * height_m

# Convert work to kilojoules
work_kJ = work_joules / 1000

# Calculate total energy intake required accounting for efficiency
total_energy_required_kJ = work_kJ / efficiency

# Calculate number of energy bars needed
number_of_bars = total_energy_required_kJ / energy_per_bar_kJ

# Round up to nearest whole number since partial bars cannot be consumed
number_of_bars_rounded = math.ceil(number_of_bars)

# Output the result
print(number_of_bars_rounded)

number_of_bars_rounded
