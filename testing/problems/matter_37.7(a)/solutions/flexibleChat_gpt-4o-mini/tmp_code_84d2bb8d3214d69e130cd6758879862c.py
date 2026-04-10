import numpy as np

# Constants
wavelength_1 = 154.433e-12  # in meters
wavelength_2 = 154.051e-12  # in meters
plane_separation = 77.8e-12  # in meters
n = 1  # first order diffraction

# Calculate sine values
sin_theta_1 = (n * wavelength_1) / (2 * plane_separation)
sin_theta_2 = (n * wavelength_2) / (2 * plane_separation)

# Error handling for arcsin
if abs(sin_theta_1) > 1 or abs(sin_theta_2) > 1:
    raise ValueError('Sine value out of range for arcsin function.')

# Calculate theta for each wavelength
theta_1 = np.arcsin(sin_theta_1)  # in radians
theta_2 = np.arcsin(sin_theta_2)  # in radians

# Calculate the difference in angles
angle_difference_radians = theta_1 - theta_2  # in radians
angle_difference_degrees = 2 * angle_difference_radians  # in radians

# Convert to degrees
angle_difference_degrees = np.degrees(angle_difference_degrees)

# Print the result
print(f'Difference in glancing angles (2θ): {angle_difference_degrees:.2f} degrees')