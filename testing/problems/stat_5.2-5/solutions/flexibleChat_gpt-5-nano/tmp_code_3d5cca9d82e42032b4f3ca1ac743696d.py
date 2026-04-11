import math
import numpy as np

# Regularized Beta CDF I_x(a, b) via a numerically stable integration
# We compute integral_0^x t^(a-1) (1-t)^(b-1) dt / B(a,b)

def beta_cdf_by_integration(u: float, a: float, b: float, N: int = 20000) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    # Transform to improve handling of singularities at t = 0 by integrating on s in [0, sqrt(u)]
    # t = s^2, dt = 2 s ds. Then integral_0^u t^(a-1) (1-t)^(b-1) dt = integral_0^{sqrt(u)} 2 s * (s^2)^(a-1) * (1 - s^2)^(b-1) ds
    sqrt_u = math.sqrt(u)
    z = np.linspace(0.0, sqrt_u, N + 1)
    # Avoid exact zero to prevent 0**negative issues
    z[0] = max(1e-12, z[0])
    t = z ** 2
    # Beta normalization constant B(a,b) computed in log-domain for stability
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    B = math.exp(logB)
    # Integrand after transformation: g(z) = 2*z * (t)^(a-1) * (1-t)^(b-1)
    g = (t ** (a - 1.0)) * ((1.0 - t) ** (b - 1.0))
    g = g * (2.0 * z)
    integral = np.trapz(g, z)
    return float(integral / B)


def find_u_by_bisection(p: float, a: float, b: float, tol: float = 1e-12, maxiter: int = 60) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        val = beta_cdf_by_integration(mid, a, b)
        if abs(val - p) < tol:
            return mid
        if val < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_f_quantile(d1: int, d2: int, p: float) -> float:
    # Try SciPy path first for accuracy when available
    try:
        from scipy.stats import f as fdist
        return float(fdist.ppf(p, d1, d2))
    except ImportError:
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        a = d1 / 2.0
        b = d2 / 2.0
        # Invert the Beta(a,b) CDF to get u, then map back to F
        u = find_u_by_bisection(p, a, b)
        fval = (d2 * u) / (d1 * (1.0 - u))
        return float(fval)


if __name__ == "__main__":
    d1, d2, p = 8, 4, 0.01
    val = compute_f_quantile(d1, d2, p)
    print("F_0.01({},{}) = {}".format(d1, d2, val))
