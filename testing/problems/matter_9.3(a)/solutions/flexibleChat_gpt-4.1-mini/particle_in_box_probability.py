# filename: particle_in_box_probability.py
import numpy as np
from scipy.integrate import quad

def psi_n(x, L, n):
    """Wavefunction for particle in 1D box at state n."""
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def probability_in_region(L, n, a, b):
    """Calculate probability of finding particle between a and b with input validation."""
    if not (0 <= a < b <= L):
        raise ValueError(f"Integration limits must satisfy 0 <= a < b <= L. Got a={a}, b={b}, L={L}.")
    integrand = lambda x: psi_n(x, L, n)**2
    prob, _ = quad(integrand, a, b)
    return prob

if __name__ == '__main__':
    # Parameters
    L = 1.0  # length of the box (arbitrary units)
    n = 1    # ground state

    # Calculate probability for 0 <= x <= L/2
    a = 0
    b = L / 2

    probability = probability_in_region(L, n, a, b)

    # Save result to a text file
    with open('probability_result.txt', 'w') as f:
        f.write(f'Probability of finding electron in 0 <= x <= L/2 for n=1: {probability:.6f}\n')

    # Print result for immediate feedback
    print(f'Probability of finding electron in 0 <= x <= L/2 for n=1: {probability:.6f}')
