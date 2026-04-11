import math
import random


def analytic_bounds(r: float, h: float, dr: float, dh: float) -> dict:
    V = (1.0/3.0) * math.pi * r * r * h
    dV_dr = (2.0/3.0) * math.pi * r * h
    dV_dh = (1.0/3.0) * math.pi * r * r
    # Conservative maximum change (sign-agnostic)
    dV_max = abs(dV_dr) * abs(dr) + abs(dV_dh) * abs(dh)
    # RMS bound for independent errors
    dV_rms = math.sqrt((dV_dr * dr) ** 2 + (dV_dh * dh) ** 2)
    V_min = V - dV_max
    V_max = V + dV_max
    relative_max_error = dV_max / V if V != 0 else float('inf')
    return {
        'V0': V,
        'dV_max': dV_max,
        'dV_rms': dV_rms,
        'V_min': V_min,
        'V_max': V_max,
        'relative_max_error': relative_max_error
    }


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


def monte_carlo_cone_volume(r, h, dr, dh, samples=100000, seed=0, distribution='uniform', std_r=None, std_h=None):
    random.seed(seed)
    volumes = []
    if distribution == 'normal':
        sr = std_r if std_r is not None else dr
        sh = std_h if std_h is not None else dh
        for _ in range(samples):
            r_p = r + random.gauss(0.0, sr)
            h_p = h + random.gauss(0.0, sh)
            if r_p <= 0 or h_p <= 0:
                continue
            volumes.append((1.0/3.0) * math.pi * r_p * r_p * h_p)
    else:
        for _ in range(samples):
            r_p = r + random.uniform(-dr, dr)
            h_p = h + random.uniform(-dh, dh)
            if r_p <= 0 or h_p <= 0:
                continue
            volumes.append((1.0/3.0) * math.pi * r_p * r_p * h_p)

    if not volumes:
        raise ValueError("No valid samples produced. Check input ranges.")

    volumes.sort()
    n = len(volumes)
    mean = sum(volumes) / n
    mean_sq = sum(v * v for v in volumes) / n
    std = math.sqrt(max(0.0, mean_sq - mean * mean))
    med = median_of_sorted(volumes)
    V0 = (1.0/3.0) * math.pi * r * r * h
    V95_lo = percentile(volumes, 0.025)
    V95_hi = percentile(volumes, 0.975)
    acceptance_rate = n / float(samples)

    return {
        'V0': V0,
        'mean': mean,
        'median': med,
        'std': std,
        'V95_lo': V95_lo,
        'V95_hi': V95_hi,
        'n_valid': n,
        'acceptance_rate': acceptance_rate
    }


def main():
    r = 10.0
    h = 25.0
    dr = 0.1
    dh = 0.1

    # Analytic differential bounds
    ana = analytic_bounds(r, h, dr, dh)
    print("Analytic differential results:")
    print("V0 = {:.6f} cm3".format(ana['V0']))
    print("dV_max = {:.6f} cm3".format(ana['dV_max']))
    print("dV_rms = {:.6f} cm3".format(ana['dV_rms']))
    print("Volume interval: [{:.6f}, {:.6f}] cm3".format(ana['V_min'], ana['V_max']))
    print("Relative max error = {:.6%}".format(ana['relative_max_error']))

    # Monte Carlo results
    res_uniform = monte_carlo_cone_volume(r, h, dr, dh, samples=200000, seed=7, distribution='uniform')
    res_normal = monte_carlo_cone_volume(r, h, dr, dh, samples=200000, seed=7, distribution='normal', std_r=dr, std_h=dh)

    print("\nMonte Carlo results:")
    print("Uniform perturbations:")
    print("  mean = {:.6f} cm3".format(res_uniform['mean']))
    print("  std  = {:.6f} cm3".format(res_uniform['std']))
    print("  95% interval: [{:.6f}, {:.6f}] cm3".format(res_uniform['V95_lo'], res_uniform['V95_hi']))
    print("  median = {:.6f} cm3".format(res_uniform['median']))
    print("  acceptance_rate = {:.4%}".format(res_uniform['acceptance_rate']))

    print("Normal perturbations:")
    print("  mean = {:.6f} cm3".format(res_normal['mean']))
    print("  std  = {:.6f} cm3".format(res_normal['std']))
    print("  95% interval: [{:.6f}, {:.6f}] cm3".format(res_normal['V95_lo'], res_normal['V95_hi']))
    print("  median = {:.6f} cm3".format(res_normal['median']))
    print("  acceptance_rate = {:.4%}".format(res_normal['acceptance_rate']))


if __name__ == '__main__':
    main()
