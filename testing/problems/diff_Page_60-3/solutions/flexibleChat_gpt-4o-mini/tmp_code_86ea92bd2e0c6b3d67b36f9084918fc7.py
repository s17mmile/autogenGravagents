import numpy as np

# Constants
initial_salt = 0  # Initial amount of salt in lb
inflow_rate = 2  # Inflow rate in gal/min
outflow_rate = 2  # Outflow rate in gal/min
salt_concentration = 0.5  # Salt concentration in lb/gal
volume = 100  # Volume of the tank in gallons

# Part 1: First 10 minutes with saltwater
# Differential equation: dS/dt = 1 - (2S/100)
# Solving for S(t)
# S(t) = 50(1 - e^(-t/50))

time_part1 = 10  # minutes
S_part1 = 50 * (1 - np.exp(-time_part1 / 50))

# Part 2: Next 10 minutes with fresh water
# Differential equation: dS/dt = - (S/50)
# S(t) = S0 * e^(-t/50)

initial_salt_after_part1 = S_part1

time_part2 = 10  # minutes
S_part2 = initial_salt_after_part1 * np.exp(-time_part2 / 50)

# Final amount of salt in the tank after 20 minutes
final_salt_amount = S_part2

# Output the result
print(f'Final amount of salt in the tank: {final_salt_amount:.3f} lb')