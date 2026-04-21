# filename: work_riemann_integral.py
import numpy as np

def riemann_sum_work_cable(weight_per_ft, shaft_depth, n_intervals, endpoint='right'):
    """Approximate work done lifting the cable using a Riemann sum."""
    delta_x = shaft_depth / n_intervals
    if endpoint == 'right':
        x = np.linspace(delta_x, shaft_depth, n_intervals)  # right endpoints
    elif endpoint == 'left':
        x = np.linspace(0, shaft_depth - delta_x, n_intervals)  # left endpoints
    elif endpoint == 'midpoint':
        x = np.linspace(delta_x/2, shaft_depth - delta_x/2, n_intervals)  # midpoints
    else:
        raise ValueError("endpoint must be 'left', 'right', or 'midpoint'")
    weights = weight_per_ft * (shaft_depth - x)
    work_approx = np.sum(weights * delta_x)
    return work_approx

def integral_work_cable(weight_per_ft, shaft_depth):
    """Calculate exact work done lifting the cable by evaluating the integral."""
    # Integral of weight_per_ft * (shaft_depth - x) dx from 0 to shaft_depth
    return weight_per_ft * (shaft_depth * shaft_depth - (shaft_depth**2) / 2)

# Constants
weight_coal = 800  # lb
weight_cable_per_ft = 2  # lb/ft
shaft_depth = 500  # ft

# Work done lifting coal (constant weight over full distance)
work_coal = weight_coal * shaft_depth

# Numerical approximation of work done lifting cable using Riemann sum
n = 1000  # number of subintervals for approximation
work_cable_approx = riemann_sum_work_cable(weight_cable_per_ft, shaft_depth, n, endpoint='right')

# Exact integral evaluation of work done lifting cable
work_cable_exact = integral_work_cable(weight_cable_per_ft, shaft_depth)

# Total work done
work_total = work_coal + work_cable_exact

# Output results
print(f"Riemann sum approximation of work done on cable (n={n}, right endpoints): {work_cable_approx:.2f} ft-lb")
print(f"Exact integral evaluation of work done on cable: {work_cable_exact:.2f} ft-lb")
print(f"Work done lifting coal: {work_coal:.2f} ft-lb")
print(f"Total work done lifting coal and cable: {work_total:.2f} ft-lb")
