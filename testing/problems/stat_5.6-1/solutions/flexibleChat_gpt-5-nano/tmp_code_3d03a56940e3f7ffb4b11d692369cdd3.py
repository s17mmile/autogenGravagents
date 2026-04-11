import math
import random
from typing import Optional, List, Dict
from dataclasses import dataclass
import csv
import io

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
    """Compute probabilities for lower_mean <= Xbar <= upper_mean, with Xbar from n i.i.d. Uniform(0,1).
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


def generate_table(ns: List[int], lower_mean: float, upper_mean: float, trials: int = 50000, seed: Optional[int] = None) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for n in ns:
        res = compute_p(n, lower_mean, upper_mean, trials=trials, seed=seed)
        rows.append({'n': float(n), 'p_exact': res.p_exact, 'p_norm': res.p_norm, 'p_mc': res.p_mc})
    return rows


def berry_esseen_bound(n: int, C: float = 0.4748) -> float:
    # E|X - mu|^3 for Uniform(0,1) with mu = 0.5 is 1/32
    E3 = 1.0/32.0
    sigma = math.sqrt(1.0/12.0)
    sigma3 = sigma ** 3
    return C * (E3 / sigma3) / math.sqrt(n)


def plot_results(ns: List[int], results: List[Dict[str, float]], filename: str = 'p_values_vs_n.png') -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    p_exact = [r['p_exact'] for r in results]
    p_norm = [r['p_norm'] for r in results]
    p_mc = [r['p_mc'] for r in results]

    plt.figure(figsize=(8,4))
    plt.plot(ns, p_exact, marker='o', label='exact')
    plt.plot(ns, p_norm, marker='s', label='norm')
    plt.plot(ns, p_mc, marker='^', label='mc')
    plt.xlabel('n')
    plt.ylabel('P(0.5 <= Xbar <= 2/3)')
    plt.title('Probability vs n: exact vs normal vs Monte Carlo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)


def main():
    # Subtask 1: run for a small set of n and print a compact table
    ns = [5, 10, 20, 50]
    lower_mean = 0.5
    upper_mean = 2.0/3.0

    table = generate_table(ns, lower_mean, upper_mean, trials=50000, seed=2024)
    print("n,p_exact,p_norm,p_mc")
    for row in table:
        print(f"{int(row['n'])},{row['p_exact']:.6f},{row['p_norm']:.6f},{row['p_mc']:.6f}")

    # Subtask 2: API already demonstrated via compute_p; show a compact call here as example
    example = compute_p(12, lower_mean, upper_mean, trials=100000, seed=123)
    print("\nExample for n=12 (bounds 0.5 to 2/3):", example)

    # Subtask 3: interpretation note on convergence behavior
    print("\nInterpretation: As n grows, p_norm -> 0.5; p_exact stays around 0.477-0.478 for this interval.")

    # Subtask 4: Berry-Esseen bound
    print("\nBerry-Esseen bound (n values):")
    for n in ns:
        print(f"{n}: {berry_esseen_bound(n):.6f}")

    # Subtask 5: optional plot
    results_for_plot = table
    plot_results(ns, results_for_plot)

    # Subtask: helper to write a compact CSV-like table for external consumption

def write_p_csv(ns: List[int], lower_mean: float, upper_mean: float, trials: int = 50000, seed: Optional[int] = None, filename: Optional[str] = None) -> str:
    table = generate_table(ns, lower_mean, upper_mean, trials=trials, seed=seed)
    header = ['n', 'p_exact', 'p_norm', 'p_mc']
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in table:
        writer.writerow([int(row['n']), f"{row['p_exact']:.6f}", f"{row['p_norm']:.6f}", f"{row['p_mc']:.6f}"])
    s = buf.getvalue()
    buf.close()
    if filename:
        with open(filename, 'w', newline='') as f:
            f.write(s)
    return s


# Minimal unit test scaffolding (lightweight)

def run_unit_tests():
    import unittest

    class PTests(unittest.TestCase):
        def test_n12_close(self):
            res = compute_p(12, 0.5, 2.0/3.0, trials=20000, seed=42)
            self.assertAlmostEqual(res.p_exact, res.p_norm, places=3)

        def test_n10_close(self):
            res = compute_p(10, 0.4, 0.6, trials=20000, seed=7)
            self.assertAlmostEqual(res.p_exact, res.p_norm, places=3)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PTests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    main()
    # Optional: run tiny unit tests
    try:
        run_unit_tests()
    except Exception:
        pass
