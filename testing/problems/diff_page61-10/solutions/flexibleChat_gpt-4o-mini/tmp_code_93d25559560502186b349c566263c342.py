import math

# Constants
monthly_payment = 800  # Monthly payment in dollars
annual_interest_rate = 0.09  # Annual interest rate (9%)
term_years = 20  # Term of the mortgage in years

# Input validation
if monthly_payment <= 0 or annual_interest_rate < 0 or term_years <= 0:
    raise ValueError('Monthly payment, interest rate, and term must be positive values.')

# Continuous interest rate
r = annual_interest_rate

# Total time in years
t = term_years

# Calculate the annual payment
annual_payment = monthly_payment * 12

# Calculate the principal using the continuous payment formula
P = annual_payment * (1 - math.exp(-r * t)) / r

# Calculate total payments over the term of the mortgage
total_payments = annual_payment * term_years

# Calculate total interest paid
total_interest = total_payments - P

# Output the total interest paid and principal amount
print(f'Total principal borrowed: ${P:.2f}')
print(f'Total interest paid during the term of the mortgage: ${total_interest:.2f}')