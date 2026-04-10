import numpy as np
from scipy.integrate import quad

# Constants
radius = 20  # radius of the pool in feet

# Depth function based on the position along the north-south axis
def depth(y):
    return (1/8) * y + 2.5  # Linear depth function

# Area of circular cross-section at fixed radius
def area():
    return np.pi * radius**2  # Area is constant based on the radius

# Volume calculation using integration
# Integrate the depth function over the height of the pool
volume = area() * quad(depth, -20, 20)[0]

# Output the volume
print(f'The volume of water in the pool is approximately {volume:.2f} cubic feet.')