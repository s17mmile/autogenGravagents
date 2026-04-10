import math

def calculate_percentage_difference(x):
    """
    Calculate the percentage difference between e^x and 1+x.
    
    Parameters:
    x (float): The value at which to evaluate the functions.
    
    Returns:
    float: The percentage difference between e^x and 1+x.
    """
    exp_x = math.exp(x)
    linear_approximation = 1 + x
    absolute_difference = abs(exp_x - linear_approximation)
    percentage_difference = (absolute_difference / abs(exp_x)) * 100
    return percentage_difference

# Constants
x = 0.0050

# Calculate and output the result
result = calculate_percentage_difference(x)
print(f'The percentage difference between e^{x} and 1+x for x={x} is: {result:.6f}%')