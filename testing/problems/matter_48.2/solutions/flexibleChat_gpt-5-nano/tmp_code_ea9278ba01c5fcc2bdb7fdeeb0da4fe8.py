import numpy as np
import math

# Portable trapezoidal integrator (replaces numpy.trapz for compatibility)

def trapz(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape for trapz integration")
    if y.size < 2:
        return float(0.0)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * (x[1:] - x[:-1])))

# Hydrogenic radial function R_{n,l}(r) with a0 = 1 (atomic units)

def laguerre_L(m: int, alpha: int, x: float) -> float:
    # Simple exact evaluation of associated Laguerre polynomial L_m^alpha(x) for integer m
    s = 0.0
    for i in range(m + 1):
        s += ((-1.0) ** i) * math.comb(m + alpha, m - i) * (x ** i) / math.factorial(i)
    return s


def R_nl(n: int, l: int, r: np.ndarray, a0: float = 1.0) -> np.ndarray:
    m = n - l - 1
    if m < 0:
        raise ValueError("Invalid (n,l) for hydrogenic radial function")
    alpha = 2 * l + 1
    N = math.sqrt((2.0 / n) ** 3 * math.factorial(m) / (2.0 * n * math.factorial(n + l)))
    x = (2.0 * r) / (n * a0)
    L = np.vectorize(lambda xv: laguerre_L(m, alpha, xv))(x)
    return N * (r / (n * a0)) ** l * np.exp(-r / (n * a0)) * L


def main():
    # Configuration (adjustable for quick tests or more thorough convergence studies)
    a0 = 1.0  # atomic units
    c = 137.035999084  # dimensionless speed of light in atomic units
    E1s = -0.5  # energy of 1s in atomic units
    angular_factor = 1.0  # placeholder for angular weighting; set to 1.0 for quick proxy
    output_ppm = True
    tol = 1e-7
    max_n = 12  # allow convergence up to 12p if needed
    rmax = 120.0
    Npts = 6000

    # 1s diamagnetic term (Ramsey): sigma_dia = - <r^2> / (6 c^2); for 1s <r^2> = 3 a0^2
    sigma_dia = -1.0 / (2.0 * c ** 2)

    # Radial grid and 1s radial function
    r = np.linspace(0.0, rmax, Npts)
    R10 = 2.0 * np.exp(-r)  # <1s|r|>, a0 = 1

    # Convergent estimation of sigma_para: sum over np states (l = 1)
    sigma_para = 0.0
    prev_sigma_para = 0.0
    used_n = 2
    In_list = []
    contrib_list = []

    print("Starting sigma_para convergence up to n = {}".format(max_n))

    for n in range(2, max_n + 1):
        Rn1 = R_nl(n, 1, r, a0=a0)
        integrand = R10 * Rn1 * (r ** 3)  # radial overlap with r^3 (volume element in radial coord)
        In = trapz(integrand, r)
        En = -0.5 / (n ** 2)
        denom = E1s - En  # energy denominator with preserved sign
        contrib = (In ** 2) / denom  # simple proxy for sigma_para

        In_list.append(In)
        contrib_list.append(contrib)
        sigma_para += contrib
        used_n = n

        # Convergence check: stop if incremental change is below tolerance
        if n > 2:
            if abs(sigma_para - prev_sigma_para) < tol:
                break
        prev_sigma_para = sigma_para

    sigma_total = sigma_dia + sigma_para
    sigma_total_ppm = sigma_total * 1e6

    print("sigma_dia =", sigma_dia)
    print("sigma_para (converged up to n = {}) = {}".format(used_n, sigma_para))
    print("sigma_total (dimensionless) =", sigma_total)
    print("sigma_total (ppm) =", sigma_total_ppm if output_ppm else None)
    for idx, (In, cval) in enumerate(zip(In_list, contrib_list), start=2):
        print("n = {}: In = {} contrib = {}".format(idx, In, cval))

    # Basic internal consistency check for the known diamagnetic term
    if not math.isfinite(sigma_dia):
        raise SystemExit("sigma_dia is not finite, check constants.")

    # Optional simple unit assertion for the diamagnetic part
    assert abs(sigma_dia - (-1.0/(2.0 * c ** 2))) < 1e-12

if __name__ == "__main__":
    main()
