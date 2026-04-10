def calculate_monthly_payment(principal, annual_interest_rate, loan_term_years):
    # Validate inputs
    if principal <= 0 or annual_interest_rate <= 0 or loan_term_years <= 0:
        raise ValueError('Principal, interest rate, and loan term must be positive values.')

    # Calculate monthly interest rate
    monthly_interest_rate = annual_interest_rate / 12  # Monthly interest rate
    # Calculate total number of payments (months)
    number_of_payments = loan_term_years * 12  # Total number of payments

    # Loan payment formula
    monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_payments) / ((1 + monthly_interest_rate) ** number_of_payments - 1)
    return monthly_payment

# Constants
principal = 8000  # Loan amount
annual_interest_rate = 0.10  # Annual interest rate
loan_term_years = 3  # Loan term in years

# Calculate and print the monthly payment
monthly_payment = calculate_monthly_payment(principal, annual_interest_rate, loan_term_years)
print(f'Monthly Payment: ${monthly_payment:.2f}')