import numpy as np
from scipy.integrate import quad

# Ensure that numpy and scipy are installed in your environment.

# Define the force dot product function based on the parameter t
# This function calculates the dot product of the force and the differential displacement vector.
def force_dot_dr(t):
    x = 2 + t  # x coordinate as a function of t
    return 3 * (x ** 2) - 12  # F(t) dot dr

# Define the limits of integration
lower_limit = 0
upper_limit = 1

# Perform the integration to calculate work done
try:
    work_done, _ = quad(force_dot_dr, lower_limit, upper_limit)
    # Print the result with context
    print(f'The total work done on the particle from (2 m, 3 m) to (3 m, 0 m) is: {work_done} J')
    # Test case to verify the result
    assert np.isclose(work_done, 7), 'The work done does not match the expected value of 7 J.'
except Exception as e:
    print(f'An error occurred during integration: {e}')