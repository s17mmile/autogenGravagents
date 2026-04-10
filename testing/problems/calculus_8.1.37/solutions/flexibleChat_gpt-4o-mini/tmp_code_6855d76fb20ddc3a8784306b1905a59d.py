import numpy as np

# Constant for the trajectory equation
GRAVITY_CONSTANT = 45

# Define the derivative of y with respect to x
def derivative_of_y(x):
    return -2 * x / GRAVITY_CONSTANT

# Define the integrand for the arc length formula
def arc_length_integrand(x):
    return np.sqrt(1 + (derivative_of_y(x))**2)

# Function to calculate the distance using Simpson's rule

def simpsons_rule_integration(func, a, b, n):
    """Approximate the integral of func from a to b using Simpson's rule with n intervals."""
    if n % 2 == 1:
        n += 1  # Ensure n is even
    h = (b - a) / n
    integral = func(a) + func(b)
    for i in range(1, n, 2):
        integral += 4 * func(a + i * h)
    for i in range(2, n-1, 2):
        integral += 2 * func(a + i * h)
    return (h / 3) * integral

# Function to calculate the distance traveled by the prey using Simpson's rule

def calculate_distance_simpson():
    x_start = 0
    x_end = 90
    n_intervals = 100  # Number of intervals for Simpson's rule
    arc_length = simpsons_rule_integration(arc_length_integrand, x_start, x_end, n_intervals)
    return round(arc_length, 1)

# Execute the calculation using Simpson's rule
if __name__ == '__main__':
    distance_simpson = calculate_distance_simpson()  # Calculate the distance using Simpson's rule
    print(f'Distance traveled by the prey (Simpson's rule): {distance_simpson} m')