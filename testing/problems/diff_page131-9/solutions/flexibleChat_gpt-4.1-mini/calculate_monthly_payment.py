# filename: calculate_monthly_payment.py

def calculate_monthly_payment(principal, annual_rate, years):
    """
    Calculate the monthly payment to pay off a loan.

    Parameters:
    principal (float): The loan amount.
    annual_rate (float): The annual interest rate (percentage).
    years (int): The loan term in years.

    Returns:
    float: The monthly payment amount.
    """
    if principal <= 0 or annual_rate < 0 or years <= 0:
        raise ValueError("Principal, annual_rate, and years must be positive numbers.")

    monthly_rate = annual_rate / 12 / 100
    total_payments = years * 12

    if monthly_rate == 0:
        # No interest case
        return principal / total_payments

    payment = (principal * monthly_rate) / (1 - (1 + monthly_rate) ** -total_payments)
    return payment

# Given values
principal = 8000
annual_interest_rate = 10
loan_term_years = 3

monthly_payment = calculate_monthly_payment(principal, annual_interest_rate, loan_term_years)

print(f"Monthly payment to pay off the loan in {loan_term_years} years: ${monthly_payment:.2f}")
