import math

"""
This script calculates the time it takes for 80% of a population to become infected based on the logistic growth model.

Constants:
- N: Total population
- initial_infected: Number of initially infected individuals
- infected_end_of_week: Number of infected individuals at the end of the week
"""

# Constants
N = 5000  # Total population
initial_infected = 160  # Initial infected
infected_end_of_week = 1200  # Infected at the end of the week

# Error handling for invalid inputs
if infected_end_of_week > N:
    raise ValueError('Infected count at the end of the week cannot exceed total population.')
if initial_infected <= 0:
    raise ValueError('Initial infected count must be greater than zero.')

# Step 1: Calculate the proportionality constant k
# Using the logistic growth equation to find k
# I(t) = N / (1 + ((N - initial_infected) / initial_infected) * e^(-kNt))

# Calculate the denominator for k
denominator = (N - initial_infected) / initial_infected

# Solve for k
k = -math.log((N / infected_end_of_week - 1) / denominator) / (N)

# Step 2: Calculate time t for 80% infection
I_target = 0.8 * N  # 80% of the population
# Using the logistic growth equation again:
# I(t) = N / (1 + ((N - initial_infected) / initial_infected) * e^(-kNt))

# Calculate time t
# Rearranging gives:
# e^(-5000kt) = ((N / I_target) - 1) / denominator
# Taking the natural log:
# -5000kt = ln(((N / I_target) - 1) / denominator)
# t = -ln(((N / I_target) - 1) / denominator) / (5000k)

t = -math.log(((N / I_target) - 1) / denominator) / (5000 * k)

# Output the time, rounded to two decimal places
print(f'Time for 80% of the population to become infected: {t:.2f} weeks')