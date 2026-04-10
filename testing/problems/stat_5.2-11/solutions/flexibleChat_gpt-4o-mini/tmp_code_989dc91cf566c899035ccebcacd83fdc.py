import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

def evaluate_integral():
    """
    Evaluates the integral:
    \[ \int_0^{0.4} \frac{\Gamma(7)}{\Gamma(4) \Gamma(3)} y^3(1-y)^2 d y \]
    """
    # Calculate the Gamma function values
    Gamma_7 = gamma(7)  # Gamma(7) = 720
    Gamma_4 = gamma(4)  # Gamma(4) = 6
    Gamma_3 = gamma(3)  # Gamma(3) = 2

    # Calculate the coefficient
    coefficient = Gamma_7 / (Gamma_4 * Gamma_3)  # 60

    # Define the integrand function
    def integrand(y):
        return y**3 * (1 - y)**2

    # Perform the integration from 0 to 0.4
    integral_value, _ = quad(integrand, 0, 0.4)

    # Multiply by the coefficient
    result = coefficient * integral_value

    return result

# Call the function and print the result
result = evaluate_integral()
print(result)