import numpy as np
from scipy.integrate import quad

# Demand function

def demand_function(x):
    """Calculate the price based on the demand function."""
    return 2000 - 0.1 * x - 0.01 * x**2

# Sales level
sales_level = 100

# Error handling for sales level
if sales_level < 0:
    raise ValueError('Sales level must be non-negative.')

# Calculate price at sales level
price_at_sales_level = demand_function(sales_level)

# Calculate total revenue
total_revenue = price_at_sales_level * sales_level

# Calculate area under the demand curve from 0 to sales_level
area_under_demand_curve, _ = quad(demand_function, 0, sales_level)

# Calculate consumer surplus
consumer_surplus = area_under_demand_curve - total_revenue

# Output the consumer surplus
print(f'Consumer Surplus: {consumer_surplus:.2f}')