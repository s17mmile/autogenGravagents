# filename: max_affordable_interest_rate.py
from scipy.optimize import bisect

def monthly_payment(rate, loan_amount, num_payments):
    """Calculate monthly payment for given monthly interest rate, loan amount, and number of payments."""
    if rate == 0:
        return loan_amount / num_payments
    return loan_amount * (rate * (1 + rate) ** num_payments) / ((1 + rate) ** num_payments - 1)

# Given parameters
loan_amount = 95000
num_years = 20
num_payments = num_years * 12
max_monthly_payment = 900

# Define function for root finding: payment - max_payment = 0
# We want to find rate such that monthly_payment(rate) = max_monthly_payment

def f(rate):
    return monthly_payment(rate, loan_amount, num_payments) - max_monthly_payment

# Interest rate bounds for bisection (monthly rate)
# Lower bound 0 (0% annual), upper bound 0.02 (2% monthly = 24% annual)
rate_lower = 0.0
rate_upper = 0.02

# Find root
max_monthly_rate = bisect(f, rate_lower, rate_upper, xtol=1e-10)

# Convert monthly rate to annual percentage rate
max_annual_rate_percent = max_monthly_rate * 12 * 100

# Output result
result_message = f"Maximum affordable annual interest rate: {max_annual_rate_percent:.4f}%"

if __name__ == "__main__":
    print(result_message)
