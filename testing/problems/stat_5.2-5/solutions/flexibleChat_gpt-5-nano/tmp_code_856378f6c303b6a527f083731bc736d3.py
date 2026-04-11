import numpy as np

try:
    from scipy.stats import f as fdist
except Exception:
    fdist = None

# parameters
d1, d2 = 8, 4
p = 0.01

if fdist is not None:
    fval = fdist.ppf(p, d1, d2)
else:
    # Fallback via Beta(d1/2, d2/2) relation: U ~ Beta(4,2), F = (d2 * U) / (d1 * (1 - U))
    # For Beta(4,2), CDF is F_U(u) = 5*u**4 - 4*u**5, so solve 5*u**4 - 4*u**5 = p
    coefs = [-4.0, 5.0, 0.0, 0.0, 0.0, -float(p)]  # -4*u^5 + 5*u^4 - p = 0
    roots = np.roots(coefs)
    real_u = [r.real for r in roots if abs(r.imag) < 1e-8 and 0.0 < r.real < 1.0]
    if real_u:
        u = real_u[0]
        fval = (d2 * u) / (d1 * (1.0 - u))
    else:
        fval = None

print("F_0.01(8,4) =", fval)
