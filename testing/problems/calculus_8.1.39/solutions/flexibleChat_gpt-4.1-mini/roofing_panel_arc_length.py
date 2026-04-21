# filename: roofing_panel_arc_length.py
import numpy as np
from scipy.integrate import quad

"""
Calculate the width of a flat metal sheet needed to produce a 28-inch corrugated roofing panel
with a sine wave profile y = sin(pi x / 7). The width corresponds to the arc length of the
sine curve over the panel width.
"""

def dydx(x):
    """Derivative of y = sin(pi x / 7)"""
    return (np.pi / 7) * np.cos((np.pi / 7) * x)

def integrand(x):
    """Integrand for arc length calculation: sqrt(1 + (dy/dx)^2)"""
    return np.sqrt(1 + dydx(x)**2)

if __name__ == "__main__":
    x_start = 0
    x_end = 28
    arc_length, error = quad(integrand, x_start, x_end)
    print(f"Width of flat metal sheet needed (arc length): {arc_length:.4f} inches")
