# filename: integrate_light_intensity.py
import numpy as np

def intensity(theta, N, d, wavelength):
    # Calculate k
    k = (np.pi * N * d * np.sin(theta)) / wavelength
    # To avoid division by zero, handle k=0 case
    # sin(k)/k approaches 1 as k->0
    # So intensity I(theta) = N^2 * (sin(k)/k)^2
    # Use np.sinc(x/pi) = sin(x)/x
    # But np.sinc is normalized as sin(pi x)/(pi x), so adjust accordingly
    # Here, sin(k)/k = np.sinc(k/np.pi)
    sinc_term = np.sinc(k / np.pi)
    I = (N**2) * (sinc_term**2)
    return I

# Given parameters
N = 10000
wavelength = 632.8e-9  # meters
d = 1e-4  # meters

# Integration limits
theta_min = -1e-6
theta_max = 1e-6

# Number of subintervals for Midpoint Rule
n = 10

# Width of each subinterval
h = (theta_max - theta_min) / n

# Midpoints
midpoints = np.linspace(theta_min + h/2, theta_max - h/2, n)

# Evaluate intensity at midpoints
I_values = intensity(midpoints, N, d, wavelength)

# Midpoint Rule approximation
integral_approx = h * np.sum(I_values)

print(f"Estimated total light intensity (integral) = {integral_approx}")
