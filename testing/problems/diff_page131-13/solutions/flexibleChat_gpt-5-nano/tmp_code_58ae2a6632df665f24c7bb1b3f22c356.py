from typing import Tuple


def payment(L: float, r: float, n: int) -> float:
    """Monthly payment for a fixed-rate loan.

    L: loan amount
    r: monthly interest rate (decimal, e.g., 0.005 for 0.5% per month)
    n: number of payments

    Returns the monthly payment P. If r <= 0, returns L / n (interest-free).
    """
    if r <= 0.0:
        return L / float(n)
    return L * r / (1.0 - (1.0 + r) ** (-float(n)))


def solve_max_monthly_rate(L: float, P_target: float, n: int, tol: float = 1e-12, max_iter: int = 200) -> Tuple[float, float]:
    """Find the maximum monthly rate r such that payment(L, r, n) <= P_target.

    Uses a safe bracketing strategy and binary search to solve payment(L, r, n) = P_target.
    Returns (r, P_at_r).
    """
    Lf = float(L)
    n = int(n)
    low = 0.0
    high = 0.2  # initial upper bound for monthly rate

    # Ensure there is a valid solution within the bracket
    if payment(Lf, low, n) > P_target:
        raise ValueError("Target payment is less than the payment at zero rate; no solution in [0, infinity).")

    while payment(Lf, high, n) < P_target:
        high *= 2.0
        if high > 1.0:
            raise ValueError("Cannot bracket root: required payment unattainable with r <= 1.0.")

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        if payment(Lf, mid, n) > P_target:
            high = mid
        else:
            low = mid
        if high - low < tol:
            break

    r = (low + high) / 2.0
    P = payment(Lf, r, n)
    return r, P


if __name__ == "__main__":
    L = 95000.0
    P_target = 900.0
    n = 240  # 20 years

    try:
        r, P = solve_max_monthly_rate(L, P_target, n)
        APR_nominal = 12.0 * r
        EAR = (1.0 + r) ** 12.0 - 1.0
        r_pct = r * 100.0
        APR_nominal_pct = APR_nominal * 100.0
        EAR_pct = EAR * 100.0

        print(f"Monthly rate r = {r:.6f} ({r_pct:.4f}% per month)")
        print(f"Monthly payment at this rate: ${P:.2f}")
        print(f"Nominal APR (monthly comp): {APR_nominal_pct:.6f}%")
        print(f"EAR: {EAR_pct:.6f}%")

        # Quick validation: payment should be within 0.01 of target
        assert abs(P - P_target) <= 0.01
    except ValueError as e:
        print(f"Solver error: {e}")