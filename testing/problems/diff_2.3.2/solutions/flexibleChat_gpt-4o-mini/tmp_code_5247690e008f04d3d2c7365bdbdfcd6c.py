import math

# Constants
P = 2000  # annual investment
r = 0.08  # annual interest rate
age_start = 25
age_end = 65

t = age_end - age_start  # number of years

# Future Value calculation for continuous contributions
FV = P * (math.exp(r * t) - 1) / r

# Output the result
print(f'The balance in the IRA at age 65 is: ${FV:.2f}')