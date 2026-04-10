import numpy as np
from scipy.optimize import bisect

try:
    from scipy.optimize import bisect
except ImportError:
    print("Error: scipy library is not installed.")
    exit(1)

def mortgage_payment(principal, interest_rate, num_payments):
    """
    Calculate the monthly mortgage payment based on the principal, interest rate, and number of payments.
    """
    if interest_rate == 0:
        return principal / num_payments  # Simple division if interest rate is zero
    monthly_rate = interest_rate / 12 / 100
    return principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)

def max_interest_rate(principal, max_payment, num_payments, tolerance=1e-6):
    """
    Calculate the maximum interest rate that allows for a specified maximum monthly payment.
    """
    def payment_difference(interest_rate):
        return mortgage_payment(principal, interest_rate, num_payments) - max_payment
    
    return bisect(payment_difference, 0, 20, xtol=tolerance)  # Interest rate between 0% and 20%

# Constants
principal = 95000  # Mortgage amount
max_payment = 900  # Maximum monthly payment
num_payments = 20 * 12  # Total number of payments (20 years)

# Calculate the maximum interest rate
max_rate = max_interest_rate(principal, max_payment, num_payments)
print(f'Maximum interest rate: {max_rate:.4f}%')