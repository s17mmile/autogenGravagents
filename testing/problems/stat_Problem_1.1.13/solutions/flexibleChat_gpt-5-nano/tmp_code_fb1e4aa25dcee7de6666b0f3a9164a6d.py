import random
from typing import Optional


def simulate(n: int = 200000, t: float = 2.0, seed: Optional[int] = None) -> float:
    """Monte Carlo simulation for the event: longer_piece >= t * shorter_piece on a unit segment.

    - n: number of trials
    - t: threshold multiplier (t >= 1.0)
    - seed: optional RNG seed for reproducibility

    Returns the fraction of trials where the condition holds.
    """
    if t < 1.0:
        raise ValueError("t must be >= 1.0")
    if seed is not None:
        random.seed(seed)

    count = 0
    for _ in range(n):
        x = random.random()  # cut point in [0, 1]
        a = x
        b = 1.0 - x
        # explicit shorter/longer calculation to avoid max/min overhead
        if a <= b:
            shorter, longer = a, b
        else:
            shorter, longer = b, a
        if longer >= t * shorter:
            count += 1
    return count / n


def analytic_probability(t: float) -> float:
    """Analytic probability for the event long >= t * short on a unit segment.

    Valid for t >= 1.0.
    """
    if t < 1.0:
        raise ValueError("t must be >= 1.0")
    return 2.0 / (t + 1.0)


def main():
    t_values = [2, 3, 4, 5]
    n = 200000

    for t in t_values:
        est = simulate(n=n, t=t)
        ana = analytic_probability(t)
        print(f"t = {t}: analytic = {ana:.6f}, simulated = {est:.6f}")

    # Demonstrate reproducibility with a fixed seed for t = 2
    seed = 42
    est = simulate(n=500000, t=2.0, seed=seed)
    ana = analytic_probability(2.0)
    print(f"(seeded) t = 2: analytic = {ana:.6f}, simulated = {est:.6f}")

    # Simple unit test for the analytic formula
    assert abs(analytic_probability(2.0) - 2.0/3.0) < 1e-12, "Analytic check failed for t = 2"


if __name__ == "__main__":
    main()
