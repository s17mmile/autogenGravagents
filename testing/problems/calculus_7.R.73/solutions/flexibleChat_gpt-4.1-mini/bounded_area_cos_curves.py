# filename: bounded_area_cos_curves.py
import sympy as sp

# Define the variable
x = sp.symbols('x')

# Define the functions
f1 = sp.cos(x)
f2 = sp.cos(x)**2

# Define the points of intersection
x1 = 0
x2 = sp.pi/2
x3 = sp.pi

# Calculate the integrals for the area
# From 0 to pi/2, cos x is above cos^2 x
integral1 = sp.integrate(f1 - f2, (x, x1, x2))
# From pi/2 to pi, cos^2 x is above cos x
integral2 = sp.integrate(f2 - f1, (x, x2, x3))

# Total bounded area
area = integral1 + integral2

# Simplify the symbolic result
area_simplified = sp.simplify(area)

# Numerical approximation
area_numeric = area.evalf()

print('Exact area bounded by the curves:', area_simplified)
print('Numerical approximation:', area_numeric)
