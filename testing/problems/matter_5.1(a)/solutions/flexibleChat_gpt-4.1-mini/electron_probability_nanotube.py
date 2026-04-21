# filename: electron_probability_nanotube.py
import math

# Constants
L = 10.0  # length of nanotube in nanometers (nm)

a = 4.95  # lower bound in nm
b = 5.05  # upper bound in nm


def probability(a, b, L):
    """
    Calculate the probability of finding the electron between positions a and b in a nanotube of length L.

    Parameters:
    a (float): lower bound position in nm (0 <= a <= L)
    b (float): upper bound position in nm (a <= b <= L)
    L (float): length of the nanotube in nm

    Returns:
    float: probability of finding the electron between a and b
    """
    if not (0 <= a <= L and 0 <= b <= L and a <= b):
        raise ValueError("Integration limits must satisfy 0 <= a <= b <= L")

    term_b = b - (L / (2 * math.pi)) * math.sin((2 * math.pi * b) / L)
    term_a = a - (L / (2 * math.pi)) * math.sin((2 * math.pi * a) / L)
    return (1 / L) * (term_b - term_a)

prob = probability(a, b, L)

# Sanity check: probability over entire length should be 1
assert abs(probability(0, L, L) - 1.0) < 1e-12, "Probability over entire length is not 1"

# Output the result
print(f"Probability of finding the electron between {a} nm and {b} nm: {prob:.6f}")
