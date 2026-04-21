# filename: find_hilltop_x_with_check.py
import sympy as sp

# Define variables
x, y = sp.symbols('x y')

# Define the system of equations from partial derivatives
# Partial derivatives:
# dz/dx = 2*y - 6*x - 18 = 0
# dz/dy = 2*x - 8*y + 28 = 0

eq1 = sp.Eq(2*y - 6*x - 18, 0)
eq2 = sp.Eq(2*x - 8*y + 28, 0)

# Solve the system for critical points
solution = sp.solve((eq1, eq2), (x, y))

# Extract the critical point coordinates
x_crit = solution[x]
y_crit = solution[y]

# Define the second derivatives for Hessian matrix
# d2z/dx2 = -6
# d2z/dy2 = -8
# d2z/dxdy = 2

# Construct Hessian matrix
H = sp.Matrix([[-6, 2],
               [2, -8]])

# Evaluate Hessian at critical point (constant here, so no substitution needed)
# Check if Hessian is negative definite for local maximum
# Negative definite if all eigenvalues are negative
eigenvals = H.eigenvals()

# Check if all eigenvalues are negative
is_local_max = all(ev < 0 for ev in eigenvals)

# Print results
print(f'Critical point at x = {x_crit}, y = {y_crit}')
print(f'Hessian eigenvalues: {list(eigenvals.keys())}')
if is_local_max:
    print('The critical point is a local maximum (top of the hill).')
else:
    print('The critical point is not a local maximum.')

# Return the x-coordinate of the top of the hill if it is a maximum
x_top = x_crit if is_local_max else None
x_top