import sympy as sp

# Define the variable
beta = sp.symbols('beta')

# Coefficients based on the derived expressions
C1 = (4 - 2 * beta) / 5
C2 = (1 + 2 * beta) / 5

# Coefficients for the quadratic equation derived from the derivative
A = C1 / 2
B = -2 * C2

# Calculate the discriminant
D = B**2 - 4 * A * 0  # C is 0 since we are looking for roots of the derivative

# Set up the condition for the discriminant to be less than or equal to zero
condition = sp.solve(D <= 0, beta)

# Display the condition
condition