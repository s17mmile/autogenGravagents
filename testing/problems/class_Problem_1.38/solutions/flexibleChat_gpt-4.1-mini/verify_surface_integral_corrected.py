# filename: verify_surface_integral_corrected.py
import numpy as np
from scipy.integrate import dblquad

# Constant curl of A = (-1, -1, -1)
curl = np.array([-1, -1, -1])

def integrand(x, y):
    z = 1 - x**2 - y**2
    if z < 0:
        return 0
    # Surface normal vector: n = (-dz/dx, -dz/dy, 1) = (2x, 2y, 1)
    normal = np.array([2*x, 2*y, 1])
    # Dot product curl . normal
    return np.dot(curl, normal)

def integrand_polar(theta, r):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Jacobian r for polar coordinates
    return integrand(x, y) * r

# Integration limits: theta from 0 to 2*pi, r from 0 to 1
result, error = dblquad(integrand_polar, 0, 2*np.pi, lambda theta: 0, lambda theta: 1)

print(f'Numerical surface integral of curl A over paraboloid surface: {result}')
print(f'Estimated error: {error}')

with open('surface_integral_result.txt', 'w') as f:
    f.write(f'Numerical surface integral of curl A over paraboloid surface: {result}\n')
    f.write(f'Estimated error: {error}\n')
