import numpy as np

# Constants
lambda_ = 632.8e-9  # Wavelength in meters
N = 10000  # Number of slits
d = 1e-4  # Distance between adjacent slits in meters

# Function to calculate intensity I(theta)
def intensity(theta):
    k = (np.pi * N * d * np.sin(theta)) / lambda_
    # Avoid division by zero
    return np.where(k != 0, (N**2) * (np.sin(k)**2) / (k**2), 0)

# Midpoint Rule parameters
a = -1e-6  # Lower limit
b = 1e-6   # Upper limit
n = 10      # Number of intervals
h = (b - a) / n  # Width of each interval

# Midpoint Rule integration
midpoints = a + (np.arange(n) + 0.5) * h  # Midpoints of the intervals
integral = np.sum(intensity(midpoints)) * h

# Output the estimated total light intensity
print(f'Total light intensity: {integral:.10f}')  # More precision in output