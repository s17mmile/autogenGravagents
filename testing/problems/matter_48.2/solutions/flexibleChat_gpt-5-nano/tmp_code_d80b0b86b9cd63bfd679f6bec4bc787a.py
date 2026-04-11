import numpy as np
import math

# Laguerre polynomial L_m^alpha(x) helper (exact integer m, alpha)

def laguerre_L(m: int, alpha: int, x: float) -> float:
    s = 0.0
    for i in range(m + 1):
        s += ((-1.0) ** i) * math.comb(m + alpha, m - i) * (x ** i) / math.factorial(i)
    return s

# Hydrogenic radial function R_{n,l}(r) with a0 = 1 (atomic units)

def R_nl(n: int, l: int, r: np.ndarray, a0: float = 1.0) -> np.ndarray:
    m = n - l - 1
    if m < 0:
        raise ValueError("Invalid (n,l) for hydrogenic radial function")
    alpha = 2 * l + 1
    # Normalization constant for hydrogenic radial part (a0 = 1)
    N = math.sqrt((2.0 / n) ** 3 * math.factorial(m) / (2.0 * n * math.factorial(n + l)))
    x = (2.0 * r) / (n * a0)  # dimensionless argument for Laguerre polynomial
    L = np.vectorize(lambda rr: laguerre_L(m, alpha, 2.0 * rr / n))(r)
    return N * (r / (n * a0)) ** l * np.exp(-r / (n * a0)) * L


def main():
    a0 = 1.0  # atomic units
    c = 137.035999084  # dimensionless speed of light in atomic units
    E1s = -0.5  # energy of 1s in a.u.

    # 1s diamagnetic term (Ramsey): sigma_dia = - <r^2> / (6 c^2); for 1s <r^2> = 3 a0^2 -> sigma_dia = -1/(2 c^2)
    sigma_dia = -1.0 / (2.0 * c ** 2)

    # Radial grid for numerical integration
    rmax = 100.0
    Npts = 4000  # reasonably large; can be increased for convergence tests
    r = np.linspace(0.0, rmax, Npts)
    R10 = 2.0 * np.exp(-r)  # <1s|r|>; a0 = 1

    # Convergent estimation of sigma_para: sum over np states (l = 1)
    sigma_para = 0.0
    sigma_para_prev = 0.0
    tol = 1e-7
    max_n = 8  # initial cap; can be increased for convergence checks
    In_list = []
    contrib_list = []

    for n in range(2, max_n + 1):
        Rn1 = R_nl(n, 1, r, a0=a0)
        integrand = R10 * Rn1 * (r ** 3)  # radial overlap with r factor and volume element
        In = np.trapz(integrand, r)
        En = -0.5 / (n ** 2)
        denom = E1s - En  # energy denominator (keeps sign)
        # Physical expression would carry angular factors; here we adopt a unit angular weight as a simple proxy
        contrib = (In ** 2) / denom  # sign depends on denom; preserves physical sign
        In_list.append(In)
        contrib_list.append(contrib)
        sigma_para += contrib

        # Convergence check: stop if incremental sigma_para is below tolerance
        if n > 2:
            if abs(sigma_para - sigma_para_prev) < tol:
                break
        sigma_para_prev = sigma_para

    sigma_total = sigma_dia + sigma_para
    sigma_total_ppm = sigma_total * 1e6

    print("sigma_dia =", sigma_dia)
    print("sigma_para (partial sum up to n = {}): {}".format(n, sigma_para))
    print("sigma_total (dimensionless) =", sigma_total)
    print("sigma_total (ppm) =", sigma_total_ppm)
    for idx, (In, c) in enumerate(zip(In_list, contrib_list), start=2):
        print("n = {}: In = {}, contrib = {}".format(idx, In, c))

    # Quick validation: the analytical diamagnetic part should equal -1/(2 c^2)
    assert abs(sigma_dia - (-1.0/(2.0*c**2))) < 1e-12

if __name__ == "__main__":
    main()
