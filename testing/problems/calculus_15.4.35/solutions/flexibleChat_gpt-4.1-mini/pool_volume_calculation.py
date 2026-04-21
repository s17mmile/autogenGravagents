# filename: pool_volume_calculation.py
import numpy as np

def depth(y):
    """Calculate depth at position y (north-south axis) in feet."""
    return (1/8)*y + 4.5

def width(y, radius=20):
    """Calculate east-west width of the pool at position y in feet."""
    # Ensure y is within the pool radius
    y = np.clip(y, -radius, radius)
    return 2 * np.sqrt(radius**2 - y**2)

def trapezoidal_integration(x, y):
    """Numerical integration using the trapezoidal rule."""
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

# Discretize y from -20 to 20 feet (north-south axis)
y_values = np.linspace(-20, 20, 1000)

# Calculate integrand values: depth(y) * width(y)
integrand = depth(y_values) * width(y_values)

# Use manual trapezoidal integration to find volume in cubic feet
volume = trapezoidal_integration(y_values, integrand)

print(f"Volume of water in the pool: {volume:.2f} cubic feet")
