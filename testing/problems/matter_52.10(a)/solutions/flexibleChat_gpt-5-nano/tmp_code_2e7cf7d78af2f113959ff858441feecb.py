import math
from typing import Tuple


def _f_x(x: float, target: float = 0.95) -> float:
    """Monotone function f(x) = x / sinh(x) - target.

    For x > 0, sinh(x) > 0, so f(x) is well-defined.
    """
    return x / math.sinh(x) - target


def _newton_solve_x(target: float = 0.95, x0: float = 0.55, tol: float = 1e-14, max_iter: int = 100) -> Tuple[float, bool]:
    """Try to solve for x in f(x) = 0 using Newton-Raphson.

    Returns (x, success). If not converged, success is False.
    """
    x = x0
    for _ in range(max_iter):
        sh = math.sinh(x)
        ch = math.cosh(x)
        if sh == 0.0:
            return x, False
        f = x / sh - target
        df = (sh - x * ch) / (sh * sh)
        if df == 0.0:
            return x, False
        dx = -f / df
        x_new = x + dx
        if abs(dx) < tol and abs(f) < tol:
            return x_new, True
        x = x_new
    return x, False


def _bisection_solve_x(target: float = 0.95, a: float = 1e-9, b: float = 2.0, tol: float = 1e-14) -> float:
    """Bracketing root finder (bisection) for f(x) = x / sinh(x) - target.

    Assumes f(a) > 0 and f(b) < 0. Returns the root x within tolerance.
    """
    def f(x: float) -> float:
        return _f_x(x, target)

    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Bisection bracket does not bracket the root. Adjust limits.")

    left, right = a, b
    while (right - left) > tol:
        mid = 0.5 * (left + right)
        fm = f(mid)
        if fm == 0:
            return mid
        if fa * fm > 0:
            left = mid
            fa = fm
        else:
            right = mid
            fb = fm
    return 0.5 * (left + right)


def solve_x_for_ratio(target: float = 0.95) -> float:
    """Solve for x in x / sinh(x) = target using a robust approach.

    Prefer Newton-Raphson for speed; fall back to bracketing if needed.
    Returns x > 0.
    """
    # Try Newton-Raphson first
    x0 = 0.55
    x, ok = _newton_solve_x(target=target, x0=x0, tol=1e-14, max_iter=100)
    if ok:
        if x > 0:
            return x
    # Fallback to bracketing
    return _bisection_solve_x(target=target, a=1e-9, b=2.0, tol=1e-14)


def temperature_from_wavenumber_cm(wavenumber_cm: float) -> Tuple[float, float, float, float]:
    """Compute the temperature T (K) such that Z_vib equals 0.95 Z_vib,approx for Br2.

    Returns (T, alpha, x, ratio) where:
      - alpha = hbar * omega / (k_B T)
      - x = alpha / 2
      - T in Kelvin
      - ratio = Z_vib / Z_vib,approx
    """
    # Physical constants (CODATA-like accuracy, in SI units)
    h = 6.62607015e-34       # J s
    c = 299792458.0            # m / s
    kB = 1.380649e-23          # J / K

    target = 0.95
    x = solve_x_for_ratio(target=target)
    alpha = 2.0 * x

    # Vibrational energy scale: E_vib = h * c * wavenumber (wavenumber in m^-1)
    wavenumber_m_inv = wavenumber_cm * 100.0  # convert cm^-1 to m^-1
    E_vib = h * c * wavenumber_m_inv

    T = E_vib / (kB * alpha)

    # Exact and approximate partition functions and their ratio for verification
    Z_vib = 1.0 / (2.0 * math.sinh(alpha / 2.0))
    Z_vib_approx = 1.0 / alpha
    ratio = Z_vib / Z_vib_approx

    return T, alpha, x, ratio


def main() -> None:
    wavenumber_cm = 323.2
    T, alpha, x, ratio = temperature_from_wavenumber_cm(wavenumber_cm)
    print("Temperature (K) for 5 percent agreement:", round(T, 6))
    print("Alpha =", alpha, ", x =", x)
    print("Z_vib / Z_vib_approx =", ratio)


if __name__ == "__main__":
    main()
