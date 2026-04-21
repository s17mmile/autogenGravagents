# filename: loan_payoff_simulation.py

# Constants
principal = 100000  # initial loan amount
annual_interest_rate = 0.09  # 9% annual interest
monthly_interest_rate = annual_interest_rate / 12  # monthly interest rate

# Payment function: payment at month t (0-based)
def payment(t):
    return 800 * (1 + t / 120)

# Initialize balance and month counter
balance = principal
month = 0

# Iterate until loan is paid off
while balance > 0:
    # Apply interest
    balance *= (1 + monthly_interest_rate)
    # Subtract payment
    balance -= payment(month)
    # Increment month
    month += 1

# Output the month when loan is fully paid
print(f"Loan will be fully paid off after {month} months.")
