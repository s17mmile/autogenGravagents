import math
from typing import Tuple, List


def constants_from_initial(y0: float, beta: float) -> Tuple[float, float]:
    """Return constants C1, C2 for the general solution y(t) = C1 * exp(t/2) + C2 * exp(-t/2)
    given initial data y(0) = y0 and y'(0) = beta.
    """
    C1 = y0 / 2.0 + beta
    C2 = y0 / 2.0 - beta
    return C1, C2


def y_solution(t: float, y0: float, beta: float) -> float:
    """Compute y(t) for the IVP 4 y'' - y = 0 with y(0) = y0, y'(0) = beta.
    The solution is y(t) = C1 e^(t/2) + C2 e^(-t/2).
    """
    C1, C2 = constants_from_initial(y0, beta)
    return C1 * math.exp(t / 2.0) + C2 * math.exp(-t / 2.0)


def analytic_decay(t: float, y0: float) -> float:
    """Analytic decaying form for the special decay case: beta = -y0/2, so y(t) = y0 * exp(-t/2)."""
    return y0 * math.exp(-t / 2.0)


def verify_initial_conditions(y0: float, beta: float) -> None:
    """Verify that y_solution(0, y0, beta) == y0 and y'(0) == beta within a tolerance.
    Raises RuntimeError if checks fail.
    """
    tol = 1e-12
    y0_calc = y_solution(0.0, y0, beta)
    C1, C2 = constants_from_initial(y0, beta)
    dy0_calc = 0.5 * (C1 - C2)
    if abs(y0_calc - y0) > tol:
        raise RuntimeError(f"Initial position check failed: y(0)={y0_calc}, expected {y0}")
    if abs(dy0_calc - beta) > tol:
        raise RuntimeError(f"Initial velocity check failed: y'(0)={dy0_calc}, expected {beta}")


def main() -> None:
    y0 = 2.0
    # Enforce the decay condition
    beta_decay = -y0 / 2.0

    # Verify initial conditions under the decay beta
    verify_initial_conditions(y0, beta_decay)

    # Display decay information
    print("beta_decay:", beta_decay)
    y0_at_0 = y_solution(0.0, y0, beta_decay)
    dy0_at_0 = 0.5 * (constants_from_initial(y0, beta_decay)[0] - constants_from_initial(y0, beta_decay)[1])
    print("Initial y(0):", y0_at_0)
    print("Initial y'(0):", dy0_at_0)

    # Compare numeric solution with analytic decay for several t values
    for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
        y_num = y_solution(t, y0, beta_decay)
        y_exp = analytic_decay(t, y0)
        diff = abs(y_num - y_exp)
        print(f"t={t:.2f}, y_solution={y_num:.6f}, analytic_decay={y_exp:.6f}, diff={diff:.6f}")
        # Gentle check: ensure the difference remains small
        if diff > 1e-12:
            raise RuntimeError(f"Decay mismatch at t={t}: numeric={y_num}, analytic={y_exp}")


if __name__ == "__main__":
    main()
