import numpy as np
from scipy.integrate import quad

# Parameters
lambda_Y1 = 1  # Rate for Y1

# Expected value of Y1
E_Y1 = 1 / lambda_Y1

# Probability density function for Y2
# Derived from the CDF of the maximum of two i.i.d. exponential random variables
# PDF(Y2) = (1 - e^(-y/2)) * e^(-y/2)
def pdf_Y2(y):
    return (1 - np.exp(-y / 2)) * np.exp(-y / 2)

# Expected value of Y2 using numerical integration
try:
    E_Y2, _ = quad(lambda y: y * pdf_Y2(y), 0, np.inf)
except Exception as e:
    print(f'Error during integration: {e}')
    E_Y2 = 0  # Fallback value in case of error

# Expected value of Z
E_Z = 2 * E_Y1 + E_Y2

# Output the result
print(f'E(Z) = {E_Z}')