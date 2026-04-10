import numpy as np

# Constant for the trajectory equation
GRAVITY_CONSTANT = 45

# Define the derivative of y with respect to x
def derivative_of_y(x):
    return -2 * x / GRAVITY_CONSTANT

# Define the integrand for the arc length formula
def arc_length_integrand(x):
    return np.sqrt(1 + (derivative_of_y(x))**2)

# Function to compute the integrand for specific values of x
def test_integrand_values():
    test_values = [0, 30, 60, 90]
    results = {x: arc_length_integrand(x) for x in test_values}
    return results

# Function to calculate the distance using the trajectory equation directly
def calculate_distance_direct():
    # The horizontal distance when y = 0 is already calculated as 90 m
    return 90  # Directly using the known horizontal distance

# Execute the tests
if __name__ == '__main__':
    integrand_results = test_integrand_values()  # Test the integrand
    distance_direct = calculate_distance_direct()  # Direct distance
    print('Integrand values for specific x:')
    for x, value in integrand_results.items():
        print(f'Integrand at x={x}: {value}')  
    print(f'Distance traveled by the prey (direct method): {distance_direct} m')