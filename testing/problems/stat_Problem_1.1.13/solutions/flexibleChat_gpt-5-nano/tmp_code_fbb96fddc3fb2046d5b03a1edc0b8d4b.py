import random
import math
from dataclasses import dataclass
from typing import Optional, Tuple


def wilson_interval(p: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Compute Wilson score interval for a proportion p with n trials at z-score.

    Returns (lower_bound, upper_bound).
    """
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


@dataclass
class MCResult:
    t: float
    n: int
    p_hat: float
    se: float
    ci_lo: float
    ci_hi: float


def simulate(n: int = 200000, t: float = 2.0, seed: Optional[int] = None, use_wilson: bool = False, z: float = 1.959963984540054) -> MCResult:
    """Monte Carlo simulation for the event: longer_piece >= t * shorter_piece on a unit segment.

    Returns an MCResult containing the estimated probability, standard error, and a confidence interval.
    Uses a local RNG to avoid side effects on the global RNG.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if t < 1.0:
        raise ValueError("t must be >= 1.0")

    rng = random.Random(seed)  # local RNG
    count = 0
    for _ in range(n):
        x = rng.random()  # cut point in [0, 1]
        a, b = x, 1.0 - x
        if a <= b:
            shorter, longer = a, b
        else:
            shorter, longer = b, a
        if longer >= t * shorter:
            count += 1

    p_hat = count / n
    se = (p_hat * (1.0 - p_hat) / n) ** 0.5

    if use_wilson:
        ci_lo, ci_hi = wilson_interval(p_hat, n, z=z)
    else:
        ci_lo = max(0.0, p_hat - z * se)
        ci_hi = min(1.0, p_hat + z * se)

    return MCResult(t=t, n=n, p_hat=p_hat, se=se, ci_lo=ci_lo, ci_hi=ci_hi)


def analytic_probability(t: float) -> float:
    if t < 1.0:
        raise ValueError("t must be >= 1.0")
    return 2.0 / (t + 1.0)


def main():
    t_values = [2, 3, 4, 5]
    n = 200000

    print("t  analytic      p_hat     ci_lo     ci_hi")
    for i, t in enumerate(t_values):
        seed = 100 + i
        res = simulate(n=n, t=t, seed=seed, use_wilson=False)
        ana = analytic_probability(t)
        print(f"{t:<2}  {ana:.6f}     {res.p_hat:.6f}  {res.ci_lo:.6f}  {res.ci_hi:.6f}")

    # Demonstrate Wilson intervals for t = 2 with larger n
    res = simulate(n=500000, t=2.0, seed=999, use_wilson=True, z=1.959963984540054)
    ana = analytic_probability(2.0)
    print(f"\nWilson CI t=2: analytic = {ana:.6f}, p_hat = {res.p_hat:.6f}, CI = ({res.ci_lo:.6f}, {res.ci_hi:.6f})")

    # Simple in-code unit check for analytic formula
    assert abs(analytic_probability(2.0) - 2.0/3.0) < 1e-12


if __name__ == "__main__":
    main()
