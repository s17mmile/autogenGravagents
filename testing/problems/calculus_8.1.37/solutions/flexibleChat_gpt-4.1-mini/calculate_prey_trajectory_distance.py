# filename: calculate_prey_trajectory_distance.py
import numpy as np
from scipy.integrate import quad


def integrand(x):
    """Calculate the integrand for arc length: sqrt(1 + (dy/dx)^2)"""
    dydx = -2 * x / 45
    return np.sqrt(1 + dydx**2)


def calculate_arc_length(x_start=0, x_end=90):
    """Calculate the arc length of the parabolic trajectory from x_start to x_end."""
    arc_length, error = quad(integrand, x_start, x_end)
    # error is the estimated absolute error of the integral
    return round(arc_length, 1)


if __name__ == '__main__':
    distance = calculate_arc_length()
    print(f"Distance traveled by the prey along the parabolic path: {distance} meters")
    distance
