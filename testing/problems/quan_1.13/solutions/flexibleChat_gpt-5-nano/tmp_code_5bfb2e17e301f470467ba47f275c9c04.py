import math
from typing import Tuple


def probability_between(x1: float, x2: float, c: float) -> float:
    """Analytic probability for |Psi(x,0)|^2 between x1 and x2.

    For t = 0, the state reduces to a form proportional to x * exp(-x^2/c^2).
    The probability density is proportional to x^2 * exp(-2 x^2 / c^2).
    With the substitution u = sqrt(2) * x / c, the closed form is:
        P = 0.5 * [erf(u2) - erf(u1)] - (1/sqrt(pi)) * [u2 e^{-u2^2} - u1 e^{-u1^2}]
    where u1 = sqrt(2) * x1 / c and u2 = sqrt(2) * x2 / c.
    """
    if x2 < x1:
        raise ValueError("x2 must be >= x1")
    if c <= 0:
        raise ValueError("c must be positive")

    u1 = math.sqrt(2.0) * x1 / c
    u2 = math.sqrt(2.0) * x2 / c

    erf_diff = math.erf(u2) - math.erf(u1)
    delta_uexp = u2 * math.exp(-u2 * u2) - u1 * math.exp(-u1 * u1)

    P = 0.5 * erf_diff - (1.0 / math.sqrt(math.pi)) * delta_uexp

    # Guard against tiny negative values due to numerical round-off
    if P < 0.0:
        P = 0.0
    elif P > 1.0:
        P = 1.0
    return P


def direct_numerical_integral(x1: float, x2: float, c: float, n: int = 2000) -> float:
    """Numerical cross-check using Simpson's rule for the same |Psi|^2 density."""
    if n % 2 == 1:
        n += 1
    if x2 < x1:
        raise ValueError("x2 must be >= x1")
    if c <= 0:
        raise ValueError("c must be positive")

    a, b = x1, x2
    h = (b - a) / n

    def integrand(x: float) -> float:
        return math.sqrt(32.0 / math.pi) * (1.0 / (c ** 3)) * (x * x) * math.exp(-2.0 * x * x / (c * c))

    s = integrand(a) + integrand(b)
    for i in range(1, n, 2):
        s += 4.0 * integrand(a + i * h)
    for i in range(2, n, 2):
        s += 2.0 * integrand(a + i * h)
    return s * h / 3.0


def main() -> None:
    c = 2.0        # Angstrom
    x1 = 2.0       # Angstrom
    x2 = 2.001     # Angstrom

    P_analytic = probability_between(x1, x2, c)
    P_numeric = direct_numerical_integral(x1, x2, c, n=2000)

    delta = abs(P_analytic - P_numeric)
    rel = delta / max(P_analytic, 1e-12)

    print("Analytic P =", P_analytic)
    print("Numeric P  =", P_numeric)
    print("Difference =", delta)
    print("Relative_error =", rel)


if __name__ == "__main__":
    main()
