# filename: numerov_harmonic_oscillator_probability_extended_domain.py
import numpy as np
from scipy.integrate import simpson

def numerov_method(V, x, E, h):
    """
    Numerov method to solve the 1D time-independent Schrodinger equation:
    psi''(x) = 2*(V(x) - E)*psi(x)
    Here, k(x) = 2*(E - V(x)) is used in the Numerov formula.
    """
    N = len(x)
    psi = np.zeros(N)
    k = 2 * (E - V)

    # Initial conditions:
    # For the ground state (v=0), wavefunction is even, so psi'(0) = 0.
    # We start integration from the left boundary x[0].
    # Set psi[0] = 0 (boundary condition at large negative x),
    # and a small nonzero psi[1] to start the recursion.
    psi[0] = 0.0
    psi[1] = 1e-5

    for i in range(1, N-1):
        psi[i+1] = ((2 * psi[i] * (1 - 5 * h**2 * k[i] / 12) - psi[i-1] * (1 + h**2 * k[i-1] / 12)) /
                    (1 + h**2 * k[i+1] / 12))

    return psi

def normalize_wavefunction(psi, x):
    """Normalize the wavefunction so that integral of |psi|^2 dx = 1."""
    norm = np.sqrt(simpson(psi**2, x))
    return psi / norm

def enforce_even_parity(psi):
    """Enforce even parity (symmetry) on the wavefunction for v=0 ground state."""
    psi_even = (psi + psi[::-1]) / 2
    return psi_even

def calculate_forbidden_probability():
    # Extend spatial grid from -10 to 10 with step 0.1
    x = np.arange(-10, 10.1, 0.1)
    h = x[1] - x[0]

    # Harmonic oscillator potential V(x) = 0.5 * x^2 (in normalized units)
    V = 0.5 * x**2

    # Energy for v=0 state: E = (v + 0.5) * hbar * omega, with hbar=omega=1
    v = 0
    E = v + 0.5

    # Compute wavefunction using Numerov method
    psi = numerov_method(V, x, E, h)

    # Enforce even parity for ground state
    psi = enforce_even_parity(psi)

    # Normalize wavefunction
    psi_norm = normalize_wavefunction(psi, x)

    # Check wavefunction decay at boundaries to ensure minimal truncation error
    if abs(psi_norm[0]) > 1e-3 or abs(psi_norm[-1]) > 1e-3:
        print("Warning: Wavefunction does not decay sufficiently at boundaries.")

    # Classical turning points where E = V(x): x = +- sqrt(2E)
    turning_point = np.sqrt(2 * E)

    # Mask for classically forbidden region (|x| > turning_point)
    forbidden_mask = (np.abs(x) > turning_point)

    # Calculate probability in forbidden region by integrating |psi|^2
    prob_forbidden = simpson(psi_norm[forbidden_mask]**2, x[forbidden_mask])

    return prob_forbidden

if __name__ == "__main__":
    prob = calculate_forbidden_probability()
    print(f"Probability of being in the classically forbidden region for v=0 state with extended domain: {prob:.6f}")
