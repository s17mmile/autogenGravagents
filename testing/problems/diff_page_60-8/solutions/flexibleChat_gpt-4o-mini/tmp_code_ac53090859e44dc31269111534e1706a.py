import math

# This program calculates the annual investment amount needed to accumulate $1,000,000 in 40 years
# at a continuous compounding interest rate of 7.5%.

# Constants
future_value = 1000000  # Future value in dollars
annual_interest_rate = 0.075  # Annual interest rate (7.5% as a decimal)
time_years = 40  # Time in years

# Error handling for invalid interest rate
if annual_interest_rate <= 0:
    raise ValueError('Interest rate must be greater than zero.')

# Calculate annual investment amount k using the rearranged formula
annual_investment = (future_value * annual_interest_rate) / (math.exp(annual_interest_rate * time_years) - 1)

# Output the result with context
print(f'To accumulate ${future_value:,} in {time_years} years at an annual interest rate of {annual_interest_rate * 100:.1f}%, the annual investment amount k needed is approximately: ${annual_investment:.2f}')