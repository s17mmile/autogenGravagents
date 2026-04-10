def calculate_monthly_payment(principal, annual_rate, years):
    # Calculate monthly mortgage payment using the fixed-rate mortgage formula
    monthly_rate = annual_rate / 12  # Monthly interest rate
    total_payments = years * 12  # Total number of payments
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**total_payments) / ((1 + monthly_rate)**total_payments - 1)
    return monthly_payment

# Constants
P = 100000  # Principal amount (loan)
annual_interest_rate_decimal = 0.09  # Annual interest rate in decimal
loan_term_years = 30  # Loan term in years

# Calculate the monthly payment
monthly_payment = calculate_monthly_payment(P, annual_interest_rate_decimal, loan_term_years)

# Output the monthly payment formatted to two decimal places
print(f'Monthly Payment: ${monthly_payment:.2f}')