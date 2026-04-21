# filename: hydrogen_electron_probability.py
import math

def probability_within_radius(a, a0=1.0):
    """
    Calculate the probability that the electron in the ground state of hydrogen
    is within radius a from the nucleus.

    Parameters:
    a (float): radius within which to find the electron probability (in units of Bohr radius)
    a0 (float): Bohr radius (default 1.0 for atomic units)

    Returns:
    float: probability value
    """
    if a < 0 or a0 <= 0:
        raise ValueError("Radius 'a' must be non-negative and Bohr radius 'a0' must be positive.")
    term = 1 + 2 * a / a0 + 2 * (a ** 2) / (a0 ** 2)
    probability = 1 - math.exp(-2 * a / a0) * term
    return probability

# Example usage:
a = 1.0  # radius in Bohr radii
prob = probability_within_radius(a)
print(f"Probability that electron is within {a} Bohr radius: {prob:.6f}")
