import numpy as np
from scipy.integrate import quad

def force(x):
    """
    Calculate the force at position x.
    
    Parameters:
    x (float): The position in feet.
    
    Returns:
    float: The force in pounds at position x.
    """
    return 5 * x**(-2)

# Limits of integration
start_point = 1  # starting point
end_point = 10   # ending point

# Calculate work done
work_done, _ = quad(force, start_point, end_point)

# Output the result with two decimal places
print(f'The work done is {work_done:.2f} foot-pounds.')