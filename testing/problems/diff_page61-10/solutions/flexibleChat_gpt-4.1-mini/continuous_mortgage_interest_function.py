# filename: continuous_mortgage_interest_function.py
import math

def calculate_total_interest(monthly_payment: float, annual_interest_rate: float, term_years: float) -> tuple:
    """
    Calculate total interest paid on a mortgage with continuous compounding and continuous payments.

    Parameters:
    monthly_payment (float): Maximum monthly payment in dollars.
    annual_interest_rate (float): Annual interest rate as a decimal (e.g., 0.09 for 9%).
    term_years (float): Term of the mortgage in years.

    Returns:
    tuple: (initial_loan_amount, total_paid, total_interest)
    """
    # Input validation
    if monthly_payment <= 0 or annual_interest_rate <= 0 or term_years <= 0:
        raise ValueError("All inputs must be positive numbers.")

    # Convert monthly payment to continuous annual payment rate (dollars per year)
    annual_payment = monthly_payment * 12

    # Calculate exponential decay term
    exp_term = math.exp(-annual_interest_rate * term_years)

    # Calculate initial loan amount (principal)
    initial_loan = (annual_payment / annual_interest_rate) * (1 - exp_term)

    # Calculate total amount paid over the term
    total_paid = annual_payment * term_years

    # Calculate total interest paid
    total_interest = total_paid - initial_loan

    return initial_loan, total_paid, total_interest


# Given parameters
monthly_payment = 800  # dollars per month
annual_interest_rate = 0.09  # 9% annual interest
term_years = 20  # mortgage term in years

# Calculate mortgage details
initial_loan, total_paid, total_interest = calculate_total_interest(monthly_payment, annual_interest_rate, term_years)

# Output results
print(f"Initial loan amount (principal): ${initial_loan:,.2f}")
print(f"Total amount paid over {term_years} years: ${total_paid:,.2f}")
print(f"Total interest paid during the term: ${total_interest:,.2f}")
