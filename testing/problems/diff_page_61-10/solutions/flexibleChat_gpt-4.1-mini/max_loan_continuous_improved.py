# filename: max_loan_continuous_improved.py
import math

def max_loan_amount(payment_monthly, annual_rate, years):
    """
    Calculate the maximum loan amount affordable with continuous compounding and continuous payments.

    Parameters:
    payment_monthly (float): Monthly payment amount in dollars (must be positive).
    annual_rate (float): Annual interest rate as a decimal (e.g., 0.09 for 9%). Must be non-negative.
    years (float): Term of the mortgage in years (must be positive).

    Returns:
    float: Maximum loan amount in dollars.
    """
    # Input validation
    if payment_monthly <= 0:
        raise ValueError("Monthly payment must be positive.")
    if annual_rate < 0:
        raise ValueError("Annual interest rate cannot be negative.")
    if years <= 0:
        raise ValueError("Term in years must be positive.")

    # Convert monthly payment to annual payment
    payment_annual = payment_monthly * 12

    r = annual_rate
    T = years

    if r == 0:
        # If interest rate is zero, present value is simply payment times term
        PV = payment_annual * T
    else:
        PV = (payment_annual / r) * (1 - math.exp(-r * T))

    return PV

# Given values
payment_monthly = 800  # dollars
annual_rate = 0.09     # 9 percent
term_years = 20        # 20 years

max_loan = max_loan_amount(payment_monthly, annual_rate, term_years)
print(f"Maximum loan amount the buyer can afford: ${max_loan:,.2f}")
