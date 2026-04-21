# filename: logistic_growth_halibut.py
import math

# Given parameters
r = 0.71  # growth rate per year
K = 80.5e6  # carrying capacity in kg
y0 = 0.25 * K  # initial biomass in kg
t = 2  # time in years

# Calculate the exponent term in the logistic formula
exponent = -r * t

# Calculate the denominator of the logistic growth formula
# (1 + ((K / y0) - 1) * e^(-r * t))
denominator = 1 + ((K / y0) - 1) * math.exp(exponent)

# Calculate biomass at time t
y_t = K / denominator

# Print the result with commas for readability
print(f"Biomass after {t} years: {y_t:,.2f} kg")
