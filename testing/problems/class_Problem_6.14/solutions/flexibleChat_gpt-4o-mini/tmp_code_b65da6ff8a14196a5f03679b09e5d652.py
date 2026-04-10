import numpy as np
from scipy.integrate import quad

# Define the parameterization of the path
# The path is parameterized by t, where t varies from 0 to 1
# theta(t) = -pi/2 + pi * t defines the angle as a function of t

def path_length(t):
    # Calculate derivatives of x, y, z with respect to t
    dx_dt = np.cos(np.pi * t) - (1 - t) * np.pi * np.sin(np.pi * t)
    dy_dt = -np.sin(np.pi * t) - (1 - t) * np.pi * np.cos(np.pi * t)
    dz_dt = 1
    # Return the integrand for the length calculation
    return np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

# Calculate the length of the path from t=0 to t=1
try:
    length, error = quad(path_length, 0, 1)
    # Output the length
    print('Length of the shortest path:', length)
except Exception as e:
    print('An error occurred during integration:', e)