# filename: calculate_diffraction_angle_difference.py
import math

def calculate_angle_difference(lambda1_pm, lambda2_pm, d_pm, order=1):
    """
    Calculate the difference in glancing angles (2 theta) for two wavelengths using Bragg's law.

    Parameters:
    lambda1_pm (float): Wavelength 1 in picometers.
    lambda2_pm (float): Wavelength 2 in picometers.
    d_pm (float): Interplanar spacing in picometers.
    order (int): Order of diffraction (default is 1).

    Returns:
    float: Difference in 2*theta angles in degrees.
    """
    sin_theta1 = order * lambda1_pm / (2 * d_pm)
    sin_theta2 = order * lambda2_pm / (2 * d_pm)

    if not (-1 <= sin_theta1 <= 1) or not (-1 <= sin_theta2 <= 1):
        raise ValueError("Invalid sine value for theta calculation. Check input values.")

    theta1_rad = math.asin(sin_theta1)
    theta2_rad = math.asin(sin_theta2)

    diff_2theta_deg = math.degrees(2 * theta1_rad - 2 * theta2_rad)
    return diff_2theta_deg

# Given values
lambda1 = 154.433  # pm
lambda2 = 154.051  # pm
d = 77.8           # pm

angle_difference = calculate_angle_difference(lambda1, lambda2, d)

with open('angle_difference_result.txt', 'w') as f:
    f.write(f'Difference in glancing angles (2 theta): {angle_difference:.6f} degrees\n')
