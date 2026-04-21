# filename: harmonic_oscillator_analytical_vs_numerov.py
import numpy as np
from scipy.integrate import simpson
from scipy.special import hermite
from math import factorial, sqrt, pi

def numerov_method(V, x, E, h):
    N = len(x)
    psi = np.zeros(N)
    k = 2 * (E - V)
    psi[0] = 0.0
    psi[1] = 1e-5
    for i in range(1, N-1):
        psi[i+1] = ((2 * psi[i] * (1 - 5 * h**2 * k[i] / 12) - psi[i-1] * (1 + h**2 * k[i-1] / 12)) /
                    (1 + h**2 * k[i+1] / 12))
    return psi

def normalize_wavefunction(psi, x):
    norm = np.sqrt(simpson(psi**2, x))
    return psi / norm

def enforce_even_parity(psi):
    return (psi + psi[::-1]) / 2

def analytical_ground_state(x):
    # Analytical ground state wavefunction for harmonic oscillator (hbar=omega=m=1)
    return (1 / (pi**0.25)) * np.exp(-x**2 / 2)

def calculate_probability_forbidden_region(psi, x, E):
    turning_point = np.sqrt(2 * E)
    forbidden_mask = (np.abs(x) > turning_point)
    prob = simpson(psi[forbidden_mask]**2, x[forbidden_mask])
    return prob

def main():
    x = np.arange(-10, 10.1, 0.1)
    h = x[1] - x[0]
    V = 0.5 * x**2
    v = 0
    E = v + 0.5

    psi_num = numerov_method(V, x, E, h)
    psi_num = enforce_even_parity(psi_num)
    psi_num = normalize_wavefunction(psi_num, x)

    psi_ana = analytical_ground_state(x)
    psi_ana = normalize_wavefunction(psi_ana, x)

    prob_num = calculate_probability_forbidden_region(psi_num, x, E)
    prob_ana = calculate_probability_forbidden_region(psi_ana, x, E)

    print(f"Numerov method forbidden region probability: {prob_num:.6f}")
    print(f"Analytical forbidden region probability: {prob_ana:.6f}")

if __name__ == '__main__':
    main()
