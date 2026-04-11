import math

# Compatibility fallback for comb (Python < 3.8)
try:
    from math import comb  # type: ignore
except ImportError:
    def comb(n, k):
        from math import factorial
        if k < 0 or k > n:
            return 0
        return factorial(n) // (factorial(k) * factorial(n - k))


def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_bounds(n=100, p=0.25, epsilon=0.05):
    """
    Compute lower bounds for P(|Y/n - p| < epsilon) where Y ~ Binomial(n, p).

    Returns a dictionary with:
    - mu: mean np
    - var: variance np(1-p)
    - t: threshold for |Y - mu| < t, where t = n * epsilon
    - k_min, k_max: integer bounds such that the event is Y in [k_min, k_max]
      (these correspond to {Y: |Y - mu| < t}). If t <= 0, the interval is empty.
    - chebyshev_bound: 1 - var / t^2 (0 if t <= 0)
    - cantelli_bound: 1 - var / (var + t^2) (0 if t <= 0)
    - normal_approx: normal approximation with continuity correction for Y in [k_min, k_max]
    - exact: exact probability sum_{k=k_min}^{k_max} C(n, k) p^k (1-p)^(n-k) (0 if empty interval)
    """

    mu = n * p
    var = n * p * (1 - p)
    t = n * epsilon

    # Edge-case: if t <= 0, the event is empty; return zeros for probabilistic bounds
    if t <= 0:
        return {
            "mu": mu,
            "var": var,
            "t": t,
            "k_min": 0,
            "k_max": -1,
            "chebyshev_bound": 0.0,
            "cantelli_bound": 0.0,
            "normal_approx": 0.0,
            "exact": 0.0,
        }

    # Derive the integer interval for the event {Y : |Y - mu| < t}
    k_min = max(0, int(math.floor(mu - t)) + 1)
    k_max = min(n, int(math.ceil(mu + t)) - 1)

    # Bounds
    chebyshev_bound = max(0.0, 1.0 - var / (t * t))
    cantelli_bound = max(0.0, 1.0 - var / (var + t * t))

    # Normal approximation with continuity correction for the interval [k_min, k_max]
    sigma = math.sqrt(var)
    if k_min > k_max or sigma <= 0:
        normal_approx = 0.0
        exact = 0.0
    else:
        lower_cont = k_min - 0.5
        upper_cont = k_max + 0.5
        z1 = (lower_cont - mu) / sigma
        z2 = (upper_cont - mu) / sigma
        normal_approx = Phi(z2) - Phi(z1)
        # Exact probability
        exact = sum(comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(k_min, k_max + 1))

    return {
        "mu": mu,
        "var": var,
        "t": t,
        "k_min": k_min,
        "k_max": k_max,
        "chebyshev_bound": chebyshev_bound,
        "cantelli_bound": cantelli_bound,
        "normal_approx": normal_approx,
        "exact": exact,
    }


if __name__ == "__main__":
    res = compute_bounds(n=100, p=0.25, epsilon=0.05)

    print("mu =", res["mu"])
    print("var =", res["var"])
    print("Interval for Y: [{}, {}]".format(res["k_min"], res["k_max"]))
    print("Chebyshev lower bound   : {:.6f}".format(res["chebyshev_bound"]))
    print("Cantelli lower bound     : {:.6f}".format(res["cantelli_bound"]))
    print("Normal-approx w/ cont. corr: {:.6f}".format(res["normal_approx"]))
    print("Exact P[{} <= Y <= {}]: {:.6f}".format(res["k_min"], res["k_max"], res["exact"]))
