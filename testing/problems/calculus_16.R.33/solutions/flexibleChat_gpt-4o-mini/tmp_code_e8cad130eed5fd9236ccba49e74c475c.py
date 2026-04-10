import numpy as np
from scipy.integrate import dblquad

# Define the curl of the vector field F

def curl_F(x, y):
    """
    Calculate the curl of the vector field F at point (x, y).
    Since we are on the plane defined by x + y + z = 1, z is expressed as 1 - x - y.
    """
    z = 1 - x - y  # Since we are on the plane x + y + z = 1
    return (z - y, -z, -x)  # Returns (curl_x, curl_y, curl_z)

# Define the integrand for the surface integral

def integrand(v, u):
    """
    Integrand function for the surface integral, extracting the second component of the curl.
    """
    curl = curl_F(u, v)
    return -curl[1]  # We only need the second component for the integral

# Define the limits for u and v
# u ranges from 0 to 1, and v ranges from 0 to 1-u

# Calculate the surface integral using double integration
result, error = dblquad(integrand, 0, 1, lambda u: 0, lambda u: 1 - u)

# Check if the integration was successful
if error < 1e-6:
    # Final result of the integral
    final_result = result
    print(f'The value of the line integral is: {final_result}')
else:
    print('Integration did not converge.')