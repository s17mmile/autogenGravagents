import math
import random
from typing import Optional, Dict
from dataclasses import dataclass

# Compatibility binomial coefficient (works on Python < 3.8 as well as >= 3.8)

def binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    try:
        from math import comb as _comb
        return _comb(n, k)
    except Exception:
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def irwin_hall_cdf(x: float, n: int) -> float:
    """CDF of S = sum_{i=1}^n U_i where U_i ~ Uniform(0,1).
    Returns F_S(x) for x in [0, n]. For x <= 0 -> 0, for x >= n -> 1.
    Uses the formula: F_S(x) = (1/n!) * sum_{k=0}^{floor(x)} (-1)^k C(n, k) (x - k)^n
    with a numerically stable summation (Kahan style).
    """
    if x <= 0:
        return 0.0
    if x >= n:
        return 1.0

    floor_x = int(math.floor(x))
    total = 0.0
    compens = 0.0  # for Kahan summation
    denom = math.factorial(n)
    for k in range(0, floor_x + 1):
        term = ((-1) ** k) * binom(n, k) * (x - k) ** n
        # Kahan summation to reduce numerical error
        y = term - compens
        t = total + y
        compens = (t - total) - y
        total = t
    return total / denom


@dataclass
class PResult:
    p_exact: float
    p_norm: float
    p_mc: float


def Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def monte_carlo_p(n: int, lower_mean: float, upper_mean: float, trials: int = 100000, seed: Optional[int] = None) -> float:
    rng = random.Random(seed)
    count = 0
    for _ in range(trials):
        s = 0.0
        for _ in range(n):
            s += rng.random()
        xbar = s / n
        if lower_mean <= xbar <= upper_mean:
            count += 1
    return count / trials


def compute_p(n: int, lower_mean: float, upper_mean: float, trials: int = 100000, seed: Optional[int] = None) -> PResult:
    """Compute probabilities for 1/2 <= Xbar <= 2/3 (or more generally lower_mean <= Xbar <= upper_mean).
    Returns a PResult with exact (Irwin-Hall), normal-approx, and Monte Carlo estimates.
    """
    if not (0.0 <= lower_mean <= upper_mean <= 1.0):
        raise ValueError("Bounds must satisfy 0.0 <= lower_mean <= upper_mean <= 1.0")

    a = lower_mean * n
    b = upper_mean * n
    p_exact = irwin_hall_cdf(b, n) - irwin_hall_cdf(a, n)

    mu_xbar = 0.5
    sd_xbar = math.sqrt((1.0/12.0) / n)
    p_norm = Phi((upper_mean - mu_xbar) / sd_xbar) - Phi((lower_mean - mu_xbar) / sd_xbar)

    p_mc = monte_carlo_p(n, lower_mean, upper_mean, trials=trials, seed=seed)

    return PResult(p_exact=p_exact, p_norm=p_norm, p_mc=p_mc)


def run_basic_tests():
    tests = [
        (12, 0.5, 2.0/3.0),
        (10, 0.4, 0.6),
        (6, 0.3, 0.8),
    ]
    for n, lo, hi in tests:
        res = compute_p(n, lo, hi, trials=50000, seed=2024)
        print(f"n={n}, lower={lo}, upper={hi} -> exact={res.p_exact:.6f}, norm={res.p_norm:.6f}, mc={res.p_mc:.6f}")


def main():
    # Preset constants (no user input): n, bounds for Xbar
    n = 12
    lower_mean = 0.5
    upper_mean = 2.0/3.0

    res = compute_p(n, lower_mean, upper_mean, trials=100000, seed=123)
    lower_bar = lower_mean * n
    upper_bar = upper_mean * n
    print(f"Exact P({lower_bar} <= S <= {upper_bar}) for n = {n}: {res.p_exact:.6f}")
    print(f"Normal approximation P({lower_mean} <= Xbar <= {upper_mean}): {res.p_norm:.6f}")
    print(f"Monte Carlo estimate: {res.p_mc:.6f}")
    print()
    run_basic_tests()


if __name__ == "__main__":
    main()
