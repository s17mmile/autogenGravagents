import sympy as sp

# Define the variable for alpha
alpha = sp.symbols('alpha')

# Calculate coefficients based on alpha
C1_coefficient = (alpha + 2) / 3
C2_coefficient = (2 * (alpha - 1)) / 3

# Find alpha such that the coefficient of e^(2t) approaches 0 as t approaches infinity
alpha_value = sp.solve(C1_coefficient, alpha)[0]

# Output the value of alpha clearly
print(f'The value of alpha that ensures the solution approaches zero as t approaches infinity is: {alpha_value}')