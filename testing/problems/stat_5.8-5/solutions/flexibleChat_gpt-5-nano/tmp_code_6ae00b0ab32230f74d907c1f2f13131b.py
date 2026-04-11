import math
from typing import Dict


def compute_bounds(n: int, p: float, eps: float, include_loose_bern: bool = True) -> Dict[str, float]:
    """Compute lower bounds for P(|Y/n - p| < eps) for Y ~ Bin(n, p).

    Returns a dict with keys:
      - 'hoeffding': Hoeffding lower bound
      - 'chebyshev': Chebyshev lower bound
      - 'bernstein_tight': Bernstein lower bound using M = max(p, 1-p)
      - 'bernstein_loose': Bernstein lower bound using M = 1.0 (optional)
    Note: bernstein_loose is always included for API stability; its numeric value is computed
    using the loose M regardless of include_loose_bern for consistent downstream usage.
    """
    # Basic input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    # Degenerate case: if p is exactly 0 or 1, Y is constant and the event holds with prob 1
    if p <= 0.0 or p >= 1.0:
        base = {
            'hoeffding': 1.0,
            'chebyshev': 1.0,
            'bernstein_tight': 1.0,
        }
        # Always include bernstein_loose for API stability; value is 1.0 as well
        base['bernstein_loose'] = 1.0
        return base

    t = eps * n              # deviation in Y units
    var = n * p * (1.0 - p)  # Var(Y)

    # Hoeffding bound
    bound_ho = 2.0 * math.exp(-2.0 * eps * eps * n)
    lb_ho = max(0.0, 1.0 - bound_ho)

    # Chebyshev bound
    bound_cheb = (p * (1.0 - p)) / (n * eps * eps)
    lb_cheb = max(0.0, 1.0 - bound_cheb)

    # Bernstein tight: M = max(p, 1-p)
    M_tight = max(p, 1.0 - p)
    den_tight = 2.0 * var + (2.0/3.0) * M_tight * t
    bound_bern_tight = 2.0 * math.exp(- (t * t) / den_tight)
    lb_bern_tight = max(0.0, 1.0 - bound_bern_tight)

    # Bernstein loose: M = 1.0
    M_loose = 1.0
    den_loose = 2.0 * var + (2.0/3.0) * M_loose * t
    bound_bern_loose = 2.0 * math.exp(- (t * t) / den_loose)
    lb_bern_loose = max(0.0, 1.0 - bound_bern_loose)

    return {
        'hoeffding': lb_ho,
        'chebyshev': lb_cheb,
        'bernstein_tight': lb_bern_tight,
        'bernstein_loose': lb_bern_loose,
    }


def main():
    n = 1000
    p = 0.25
    eps = 0.05

    bounds = compute_bounds(n, p, eps, include_loose_bern=True)
    fmt = ".6f"
    print("Hoeffding lower bound:", format(bounds['hoeffding'], fmt))
    print("Chebyshev lower bound:", format(bounds['chebyshev'], fmt))
    print("Bernstein (tight) lower bound:", format(bounds['bernstein_tight'], fmt))
    print("Bernstein (loose) lower bound:", format(bounds['bernstein_loose'], fmt))


if __name__ == "__main__":
    main()
