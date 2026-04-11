from typing import List, Tuple, Optional

# Constants (consistent with the problem statement)
R = 0.082057338  # L atm mol^-1 K^-1
T = 300.0          # K
A = 1.352          # L^2 atm mol^-2 (van der Waals a parameter)
B = 0.0387          # L mol^-1 (van der Waals b parameter)


def f_V(V: float, P: float, R: float = R, T: float = T, A: float = A, B: float = B) -> float:
    """van der Waals residual for a given V at pressure P.
    Returns RT/(V - B) - A/V^2 - P
    """
    return R * T / (V - B) - A / (V ** 2) - P


def bisect_root(func, a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> Optional[float]:
    """Bisection root finder for func on [a, b]. Returns None if no sign change."""
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        return None
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = func(c)
        if abs(fc) < 1e-14 or (b - a) < tol:
            return c
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def solve_V(P: float, R: float = R, T: float = T, A: float = A, B: float = B) -> float:
    """Solve the van der Waals equation for molar volume V at pressure P.
    Returns the largest real root (gas-like branch). Falls back to ideal gas volume if no root found."""
    V_min = B * 1.0000001 + 1e-12
    V_ideal = R * T / P
    V_max = max(V_ideal * 5.0, 0.1)

    func = lambda V: f_V(V, P, R, T, A, B)

    # Bracket potential roots by scanning a dense grid in V
    Ngrid = 2000
    V_prev = V_min
    f_prev = func(V_prev)
    roots: List[float] = []

    for i in range(1, Ngrid + 1):
        t = i / Ngrid
        V = V_min + t * (V_max - V_min)
        f = func(V)
        if f_prev == 0.0:
            roots.append(V_prev)
        if f_prev * f < 0.0:
            r = bisect_root(func, V_prev, V, tol=1e-12, max_iter=200)
            if r is not None:
                roots.append(r)
        V_prev, f_prev = V, f

    if not roots:
        # Fallback to ideal gas volume if no root found
        return V_ideal

    # Keep the largest root (gas-like branch) and deduplicate near-equal roots
    roots_sorted = sorted(roots)
    unique_roots: List[float] = []
    for rv in roots_sorted:
        if not unique_roots or abs(rv - unique_roots[-1]) > 1e-9:
            unique_roots.append(rv)
    V_gas = max(unique_roots)
    return V_gas


def dVdT_at_P(P: float, R: float = R, T: float = T, A: float = A, B: float = B) -> Tuple[float, float]:
    """Compute V(P) and (dV/dT)_P for the given P using implicit differentiation."""
    V = solve_V(P, R, T, A, B)
    dF_dT = R / (V - B)
    dF_dV = -R * T / (V - B) ** 2 + 2.0 * A / (V ** 3)
    if abs(dF_dV) < 1e-18:
        dVdT = 0.0
    else:
        dVdT = -dF_dT / dF_dV
    return V, dVdT


def dHdP_at_P(P: float, R: float = R, T: float = T, A: float = A, B: float = B) -> float:
    """Compute (dH/dP)_T at fixed T for pressure P."""
    V, dVdT = dVdT_at_P(P, R, T, A, B)
    return V - T * dVdT


def simpson_integration(x: List[float], y: List[float]) -> float:
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals (n even).")
    h = x[1] - x[0]
    s = y[0] + y[-1]
    s += 4.0 * sum(y[i] for i in range(1, n, 2))
    s += 2.0 * sum(y[i] for i in range(2, n, 2))
    return s * h / 3.0


def compute_delta_Hm(P_start: float = 500.0, P_end: float = 1.0, N: int = 600) -> Tuple[float, float]:
    """Integrate (dH/dP)_T from P_start to P_end at fixed T to obtain Delta H_m.
    Returns (Delta H_m in J/mol, Delta H_m in L atm/mol).
    """
    if N % 2 != 0:
        N += 1
    # Build ascending pressure grid from P_end to P_start for Simpson's rule, then negate
    step = (P_start - P_end) / N
    P_vals = [P_end + i * step for i in range(N + 1)]
    f_vals: List[float] = []
    for P in P_vals:
        f_vals.append(dHdP_at_P(P))

    # integral of (dH/dP)_T dP from P_end to P_start (ascending)
    integral_asc = simpson_integration(P_vals, f_vals)
    delta_H_L_atm = -integral_asc  # convert to path from 500 atm down to 1 atm
    delta_H_J_per_mol = delta_H_L_atm * 101.325
    return delta_H_J_per_mol, delta_H_L_atm


def main():
    V_500 = solve_V(500.0)
    V_50 = solve_V(50.0)
    V_1 = solve_V(1.0)

    delta_H_J, delta_H_L = compute_delta_Hm(P_start=500.0, P_end=1.0, N=600)

    print("End-state molar volumes (L/mol):")
    print("  at 500 atm: {:.6f}".format(V_500))
    print("  at 50 atm : {:.6f}".format(V_50))
    print("  at 1 atm  : {:.6f}".format(V_1))
    print()
    print("Delta H_m (P1 = 500 atm -> P2 = 1 atm, T = 300 K):")
    print("  Delta H_m = {:.6f} J/mol = {:.6f} kJ/mol".format(delta_H_J, delta_H_J / 1000.0))


if __name__ == "__main__":
    main()
