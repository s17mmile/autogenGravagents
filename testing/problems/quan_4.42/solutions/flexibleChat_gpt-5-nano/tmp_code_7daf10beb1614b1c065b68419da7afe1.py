import numpy as np
import math

"""
Numerov solver for the 1D harmonic oscillator ground state (v = 0).
Features:
- Left boundary initialization with a second-order correction for improved accuracy
- Numerov sweep on a symmetric domain [-L, L]
- Normalization via trapezoidal rule
- Classically forbidden region probability computed with trapezoidal integration
- Comparison to analytic erfc(x_turn) in the chosen units

All constants are preset; the function returns the wavefunction psi, grid x,
forbidden probability P_forbidden, and the analytic reference P_analytic.
"""

def numerov_solve(m=1.0, hbar=1.0, omega=1.0, L=5.0, dx=0.1):
    x = np.arange(-L, L + dx, dx)
    N = len(x)

    # Potential and energy for harmonic oscillator
    V = 0.5 * m * (omega**2) * (x**2)
    E = 0.5 * hbar * omega

    # Numerov auxiliary function f(x) = (2m/hbar^2) * (V - E)
    f = (2.0 * m / (hbar**2)) * (V - E)
    h = dx

    # Left boundary initialization with second-order correction
    x0 = x[0]
    y0 = np.exp(-0.5 * (x0**2))  # asymptotic Gaussian near left edge
    ydot0 = -x0 * y0             # y'(x0) for Gaussian
    f0 = f[0]
    y1 = y0 + dx * ydot0 + 0.5 * (dx**2) * f0 * y0  # second-order correction

    psi = np.zeros(N, dtype=float)
    psi[0] = y0
    if N > 1:
        psi[1] = y1

    # Numerov sweep: n = 1 ... N-2
    for n in range(1, N - 1):
        f_n = f[n]
        f_nm1 = f[n - 1]
        f_np1 = f[n + 1]
        coeff_n = 1.0 - (5.0 * h**2 / 12.0) * f_n
        coeff_nm1 = 1.0 + (h**2 / 12.0) * f_nm1
        coeff_np1 = 1.0 + (h**2 / 12.0) * f_np1
        psi[n + 1] = (2.0 * psi[n] * coeff_n - psi[n - 1] * coeff_nm1) / coeff_np1

    # Normalization using trapezoidal rule for robustness
    norm = np.sqrt(np.trapz(psi**2, x))
    psi /= norm

    # Turning point and forbidden region probability
    x_turn = (hbar / (m * omega))**0.5  # should be 1.0 in these units
    mask_forbidden = np.abs(x) > x_turn
    P_forbidden = np.trapz(psi[mask_forbidden]**2, x[mask_forbidden])
    P_analytic = math.erfc(x_turn)

    return psi, x, P_forbidden, P_analytic


def main():
    psi, x, P_forbidden, P_analytic = numerov_solve()

    print("P_forbidden numerov =", P_forbidden)
    print("P_analytic erfc(1) =", P_analytic)

    # Simple parity check: compare psi(-x) to psi(x) for symmetry (within tolerance)
    # This is a basic sanity check for the symmetric HO ground state.
    symmetry_err = np.max(np.abs(psi + psi[::-1]))  # since psi is even, psi(-x) ~ psi(x) -> difference with sign
    # For even function, psi(x) - psi(-x) should be ~0; we approximate with max abs(psi - psi[::-1])
    symmetry_err2 = np.max(np.abs(psi - psi[::-1]))
    print("Symmetry error (psi(x) vs psi(-x)) approx2:", symmetry_err2)

if __name__ == "__main__":
    main()
