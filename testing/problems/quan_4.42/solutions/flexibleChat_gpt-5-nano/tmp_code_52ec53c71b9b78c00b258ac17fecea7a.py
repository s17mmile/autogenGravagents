import numpy as np
import math
from dataclasses import dataclass

"""
Generalized Numerov-based solver for 1D Schrodinger equation with a flexible
potential interface. Includes a left-boundary second-order initialization,
portable trapezoidal integration, domain adequacy checks, and a structured
return object for easy testing and reuse.

Defaults: m = 1, hbar = 1, omega = 1, L = 5, dx = 0.1. HO potential is used by
default; V_func and non-HO potentials require explicit E.
"""

def trapz(y, x):
    y = np.asarray(y)
    x = np.asarray(x)
    if y.size != x.size:
        raise ValueError("y and x must have the same length for integration")
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    avg = (y[:-1] + y[1:]) * 0.5
    return float(np.sum(avg * dx))

@dataclass
class NumerovResult:
    psi: np.ndarray
    x: np.ndarray
    P_forbidden: float
    P_analytic: float
    norm: float
    domain_ok: bool
    dx: float
    L: float
    V_func_is_ho: bool
    E_used: float
    normalization_error: float
    parity_error: float
    left_edge: float
    right_edge: float


def numerov_solve(m=1.0, hbar=1.0, omega=1.0, L=5.0, dx=0.1, V_func=None, E=None, tail_type='ho-tail', domain_tol=1e-6):
    x = np.arange(-L, L + dx, dx)
    N = len(x)

    # Potential handling: HO by default
    if V_func is None:
        is_ho = True
        V = 0.5 * m * (omega**2) * (x**2)
        E_used = E if E is not None else 0.5 * hbar * omega
    else:
        is_ho = False
        V = V_func(x)
        if E is None:
            raise ValueError("When using a custom V_func, you must provide the energy E.")
        E_used = E

    # Numerov auxiliary function f(x) = (2m/hbar^2) * (V - E)
    f = (2.0 * m / (hbar**2)) * (V - E_used)
    h = dx

    # Left boundary initialization
    if is_ho:
        x0 = x[0]
        y0 = np.exp(-0.5 * (x0**2))  # HO tail
        ydot0 = -x0 * y0             # y'(x0) for HO tail
        f0 = f[0]
        y1 = y0 + dx * ydot0 + 0.5 * (dx**2) * f0 * y0  # second-order correction
    else:
        x0 = x[0]
        # Generic decaying seed for non-HO potentials; small and smooth
        y0 = 1e-6
        ydot0 = -x0 * y0
        f0 = f[0]
        y1 = y0 + dx * ydot0 + 0.5 * (dx**2) * f0 * y0

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

    # Normalization using trapezoidal rule
    integral = trapz(psi**2, x)
    norm = math.sqrt(integral)
    psi /= norm

    # Turning point and forbidden region probability
    x_turn = (hbar / (m * omega))**0.5  # HO turning point is 1 in default units
    mask_forbidden = np.abs(x) > x_turn
    P_forbidden = trapz(psi[mask_forbidden]**2, x[mask_forbidden])
    P_analytic = math.erfc(x_turn)

    left_edge = abs(psi[0]); right_edge = abs(psi[-1])
    domain_ok = (left_edge < domain_tol) and (right_edge < domain_tol)
    parity_error = float(np.max(np.abs(psi - psi[::-1])))
    normalization_error = abs(integral - 1.0)

    return NumerovResult(
        psi=psi,
        x=x,
        P_forbidden=P_forbidden,
        P_analytic=P_analytic,
        norm=norm,
        domain_ok=domain_ok,
        dx=dx,
        L=L,
        V_func_is_ho=is_ho,
        E_used=E_used,
        normalization_error=normalization_error,
        parity_error=parity_error,
        left_edge=left_edge,
        right_edge=right_edge,
    )


def main():
    # Base HO case
    data = numerov_solve()
    print("P_forbidden =", data.P_forbidden)
    print("P_analytic =", data.P_analytic)
    print("domain_ok =", data.domain_ok)
    print("dx =", data.dx, " L =", data.L)
    print("normalization_error =", data.normalization_error)
    print("parity_error =", data.parity_error)

    # Convergence study over dx
    print("\nConvergence study over dx:")
    for dx in [0.2, 0.1, 0.05, 0.02]:
        d = numerov_solve(dx=dx)
        print("dx={:.3f}  P_forbidden={:.6f}  norm={:.6f}  norm_err={:.2e}".format(
            dx, d.P_forbidden, d.norm, d.normalization_error))

    # Domain adequacy and parity sanity checks for the base run
    print("\nDomain adequacy checks:")
    print("Left edge =", data.left_edge, " Right edge =", data.right_edge)
    print("Parity error =", data.parity_error)

if __name__ == "__main__":
    main()
