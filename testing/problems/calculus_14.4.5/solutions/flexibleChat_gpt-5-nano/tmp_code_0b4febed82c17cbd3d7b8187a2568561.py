import math
import random
from dataclasses import dataclass


def percentile(sorted_data, p):
    if not sorted_data:
        raise ValueError("No data to compute percentile.")
    if p <= 0:
        return sorted_data[0]
    if p >= 1:
        return sorted_data[-1]
    n = len(sorted_data)
    pos = (n - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    w = pos - lo
    return (1.0 - w) * sorted_data[lo] + w * sorted_data[hi]


def median_of_sorted(sorted_data):
    n = len(sorted_data)
    if n == 0:
        raise ValueError("No data to compute median.")
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return 0.5 * (sorted_data[n // 2 - 1] + sorted_data[n // 2])


@dataclass
class MCResult:
    V0: float
    mean: float
    median: float
    std: float
    V95_lo: float
    V95_hi: float
    n_valid: int
    acceptance_rate: float


def monte_carlo_cone_volume(r, h, dr, dh, samples=100000, seed=0, distribution='uniform', std_r=None, std_h=None) -> MCResult:
    random.seed(seed)
    volumes = []
    invalid = 0

    if distribution == 'normal':
        sr = std_r if std_r is not None else dr
        sh = std_h if std_h is not None else dh
        for _ in range(samples):
            r_p = r + random.gauss(0.0, sr)
            h_p = h + random.gauss(0.0, sh)
            if r_p <= 0 or h_p <= 0:
                invalid += 1
                continue
            volumes.append((1.0/3.0) * math.pi * r_p * r_p * h_p)
    else:
        for _ in range(samples):
            r_p = r + random.uniform(-dr, dr)
            h_p = h + random.uniform(-dh, dh)
            if r_p <= 0 or h_p <= 0:
                invalid += 1
                continue
            volumes.append((1.0/3.0) * math.pi * r_p * r_p * h_p)

    n = len(volumes)
    if n == 0:
        raise ValueError("No valid samples produced. Check input ranges.")

    volumes.sort()
    mean = sum(volumes) / n
    mean_sq = sum(v * v for v in volumes) / n
    std = math.sqrt(max(0.0, mean_sq - mean * mean))
    med = median_of_sorted(volumes)
    V0 = (1.0/3.0) * math.pi * r * r * h
    V95_lo = percentile(volumes, 0.025)
    V95_hi = percentile(volumes, 0.975)
    acceptance_rate = n / float(samples)

    return MCResult(V0, mean, med, std, V95_lo, V95_hi, n, acceptance_rate)


def main():
    r = 10.0
    h = 25.0
    dr = 0.1
    dh = 0.1

    # Uniform perturbation model
    res_uniform = monte_carlo_cone_volume(r, h, dr, dh, samples=200000, seed=123, distribution='uniform')
    print("Base V0 = {:.6f} cm3".format(res_uniform.V0))
    print("Uniform: mean = {:.6f} cm3, std = {:.6f} cm3".format(res_uniform.mean, res_uniform.std))
    print("Uniform 95% interval: [{:.6f}, {:.6f}] cm3".format(res_uniform.V95_lo, res_uniform.V95_hi))
    print("Uniform median = {:.6f} cm3".format(res_uniform.median))
    print("Uniform acceptance_rate = {:.4%}, n_valid = {}".format(res_uniform.acceptance_rate, res_uniform.n_valid))

    # Normal perturbation model
    res_normal = monte_carlo_cone_volume(r, h, dr, dh, samples=200000, seed=123, distribution='normal', std_r=dr, std_h=dh)
    print("\nNormal: mean = {:.6f} cm3, std = {:.6f} cm3".format(res_normal.mean, res_normal.std))
    print("Normal 95% interval: [{:.6f}, {:.6f}] cm3".format(res_normal.V95_lo, res_normal.V95_hi))
    print("Normal median = {:.6f} cm3".format(res_normal.median))
    print("Normal acceptance_rate = {:.4%}, n_valid = {}".format(res_normal.acceptance_rate, res_normal.n_valid))


if __name__ == '__main__':
    main()