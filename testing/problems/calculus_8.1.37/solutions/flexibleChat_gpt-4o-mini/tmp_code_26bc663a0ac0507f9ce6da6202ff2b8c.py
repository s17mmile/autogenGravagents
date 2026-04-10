import numpy as np
from scipy.integrate import quad

# Constant for the trajectory equation
GRAVITY_CONSTANT = 45

# Define the derivative of y with respect to x
def derivative_of_y(x):
    return -2 * x / GRAVITY_CONSTANT

# Define the integrand for the arc length formula
def arc_length_integrand(x):
    return np.sqrt(1 + (derivative_of_y(x))**2)

# Function to calculate the distance traveled by the prey

def calculate_distance():
    x_start = 0
    x_end = 90
    arc_length, _ = quad(arc_length_integrand, x_start, x_end)
    return round(arc_length, 1)

# Execute the calculation
if __name__ == '__main__':
    distance = calculate_distance()  # Calculate the distance
    print(f'Distance traveled by the prey: {distance} m')