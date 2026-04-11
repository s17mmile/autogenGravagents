import numpy as np

def solve_vdw_vm(T, P_bar, a, b, R=0.08314):
    """
    Solve the van der Waals equation for the molar volume V_m given T, P_bar, a, b, and the gas constant R.
    Returns the physically meaningful V_m in L/mol.

    van der Waals cubic form (in V):
        P V^3 - (P b + R T) V^2 + a V - a b = 0

    Procedure:
    - Solve the cubic for V using numpy.roots
    - Collect real roots and filter those > b
    - For each candidate, compute P_calc from EOS and choose the root with the smallest residual |P_calc - P_bar|
      In case of ties, choose the largest root (gas-like state in many regions).
    """
    RT = R * T
    coeffs = [P_bar, -(P_bar * b + RT), a, -a * b]
    roots = np.roots(coeffs)

    tol_imag = 1e-8
    real_roots = [r.real for r in roots if abs(r.imag) < tol_imag]
    candidates = [v for v in real_roots if v > b]

    if not candidates:
        raise ValueError("No real root above b found for the given P, T, a, b.")

    # Evaluate residuals for each candidate and pick the best
    best_vm = None
    best_res = float('inf')
    residuals = {}
    for Vm in candidates:
        P_calc = (R * T) / (Vm - b) - a / (Vm**2)
        res = abs(P_calc - P_bar)
        residuals[Vm] = res
        if res < best_res:
            best_res = res
            best_vm = Vm

    # If multiple roots have nearly identical residuals, prefer the largest Vm (gas-like)
    close = [Vm for Vm, res in residuals.items() if abs(res - best_res) <= 1e-6]
    if close:
        best_vm = max(close)

    return best_vm


def main():
    # Given conditions for CO2
    T = 500.0          # K
    P_atm = 100.0      # atm
    P_bar = P_atm * 1.01325  # convert atm to bar
    a = 3.59           # bar*L^2/mol^2
    b = 0.0427         # L/mol
    R = 0.08314          # L*bar/(mol*K)

    Vm = solve_vdw_vm(T, P_bar, a, b, R)
    V_ideal = (R * T) / P_bar
    percent_diff = (V_ideal - Vm) / V_ideal * 100.0

    print("Van der Waals Vm =", Vm, "L/mol")
    print("Ideal Vm =", V_ideal, "L/mol")
    print("Percent difference =", percent_diff, "%")


if __name__ == '__main__':
    main()
