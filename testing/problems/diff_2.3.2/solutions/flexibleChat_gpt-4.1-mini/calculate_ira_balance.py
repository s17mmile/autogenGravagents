# filename: calculate_ira_balance.py
import math

def calculate_ira_balance(payment_rate, interest_rate, years):
    """
    Calculate the future value of continuous contributions with continuous compounding interest.

    Parameters:
    payment_rate (float): Continuous annual investment amount.
    interest_rate (float): Annual interest rate as a decimal.
    years (float): Number of years the money is invested.

    Returns:
    float: Future value of the investment.
    """
    if payment_rate < 0 or interest_rate < 0 or years < 0:
        raise ValueError("All input parameters must be non-negative.")
    
    if interest_rate == 0:
        # If interest rate is zero, future value is simply payment_rate * years
        return payment_rate * years
    
    fv = payment_rate * (math.exp(interest_rate * years) - 1) / interest_rate
    return fv

# Given parameters
payment_rate = 2000  # continuous annual investment in dollars
interest_rate = 0.08  # annual interest rate (8%)
years = 65 - 25  # investment period in years

# Calculate future value
future_value = calculate_ira_balance(payment_rate, interest_rate, years)

# Print the result
print(f"The balance in the IRA at age 65 will be approximately ${future_value:,.2f}.")
