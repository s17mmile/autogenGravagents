import math

# Function to calculate time for investment growth

def calculate_investment_time(initial_investment, final_amount, annual_rate):
    if initial_investment <= 0:
        raise ValueError('Initial investment must be greater than zero.')
    return math.log(final_amount / initial_investment) / annual_rate

# Constants
annual_rate = 0.07  # Annual interest rate (7%)

# User input for investment amounts
initial_investment = 1000  # Example initial investment amount
final_amount = 2000        # Example final amount

# Calculate T
try:
    T = calculate_investment_time(initial_investment, final_amount, annual_rate)
    # Output the result
    print(f'The time T required for the investment to grow from {initial_investment} to {final_amount} at an annual rate of {annual_rate * 100}% is approximately {T:.2f} years.')
except ValueError as e:
    print(e)