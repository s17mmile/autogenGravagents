# filename: consumer_surplus_calculation.py
import numpy as np
from scipy.integrate import quad

# Consumer surplus calculation for demand function p = 2000 - 0.1x - 0.01x^2
# Consumer surplus is the area under the demand curve minus total revenue at given sales level

def demand_function(x):
    return 2000 - 0.1*x - 0.01*x**2

sales_level = 100

# Calculate price at sales level
price_at_100 = demand_function(sales_level)

# Calculate integral of demand function from 0 to sales_level
integral_value, _ = quad(demand_function, 0, sales_level)

# Calculate total revenue
total_revenue = price_at_100 * sales_level

# Calculate consumer surplus
consumer_surplus = integral_value - total_revenue

print(f"Consumer Surplus at sales level {sales_level}: {consumer_surplus:.2f}")
