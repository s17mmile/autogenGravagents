# filename: continuous_investment_calculation.py
import math

def calculate_continuous_investment(A, r, t):
    """
    Calculate the continuous investment rate k needed to accumulate amount A
    over time t years with continuous compounding interest rate r.

    Parameters:
    A (float): Target amount in dollars
    r (float): Annual interest rate (as a decimal, e.g., 0.075 for 7.5%)
    t (float): Time in years

    Returns:
    float: Required continuous investment rate k (dollars per year)
    """
    return (A * r) / (math.exp(r * t) - 1)

# Given parameters
A = 1_000_000  # target amount in dollars
r = 0.075      # annual interest rate (7.5%)
t = 40         # time in years

k = calculate_continuous_investment(A, r, t)
print(f"Required continuous investment rate k: ${k:.2f} per year")
