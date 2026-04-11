import math

# Analytic crossing time with stability near y -> 4

def analytic_T(y0: float, y_target: float) -> float:
    """Return the analytic time T when the solution first reaches y_target.
    Assumes 0 < y0 < 4 and 0 < y_target < 4, and y_target > y0 for upward crossing.
    Uses: y/(4 - y) = K * exp((2/3) t^2), with K = y0/(4 - y0).
    Solving for T^2 gives T^2 = (3/2) * ln( (y_target/(4 - y_target)) / (y0/(4 - y0)) ).
    To avoid numerical cancellation near y_target ~ 4, log1p is used when the ratio is near 1.
    """
    if not (0.0 < y0 < 4.0):
        raise ValueError("y0 must be strictly between 0 and 4")
    if not (0.0 < y_target < 4.0):
        raise ValueError("y_target must be strictly between 0 and 4")
    if y_target <= y0:
        raise ValueError("y_target must be greater than y0 for an upward crossing")

    z = y_target / (4.0 - y_target)
    K = y0 / (4.0 - y0)
    ratio = z / K
    if ratio <= 0.0:
        raise ValueError("Invalid ratio for logarithm in analytic time computation")

    # Use log1p for numerical stability when ratio is close to 1
    if abs(ratio - 1.0) < 1e-12:
        T2 = (3.0/2.0) * math.log1p(ratio - 1.0)
    else:
        T2 = (3.0/2.0) * math.log(ratio)

    return math.sqrt(T2)


def f(t: float, y: float) -> float:
    """Right-hand side of the ODE dy/dt = (t/3) y (4 - y)."""
    return (t / 3.0) * y * (4.0 - y)


def rk4_crossing_time(y0: float, y_target: float, dt: float = 4e-4, max_t: float = 10.0) -> float:
    """Numerically estimate the first crossing time using RK4 with a final-step
    linear interpolation. Returns None if no crossing occurs within max_t."""
    t = 0.0
    y = y0
    t_prev, y_prev = t, y

    while t < max_t and y < y_target:
        t_prev, y_prev = t, y
        k1 = f(t, y)
        k2 = f(t + dt/2.0, y + dt * k1/2.0)
        k3 = f(t + dt/2.0, y + dt * k2/2.0)
        k4 = f(t + dt, y + dt * k3)
        y = y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t = t + dt
        if y >= y_target:
            dy = y - y_prev
            if dy != 0.0:
                alpha = (y_target - y_prev) / dy
            else:
                alpha = 0.0
            return t_prev + dt * alpha
    return None


def main():
    # Preset problem constants (as in earlier discussion)
    y0 = 0.5
    y_target = 3.98

    # Analytic time (with stability considerations)
    T_analytic = analytic_T(y0, y_target)

    # Numerical cross-check via RK4
    T_numerical = rk4_crossing_time(y0, y_target, dt=4e-4, max_t=10.0)

    print("Analytic T = {:.6f}".format(T_analytic))
    if T_numerical is None:
        print("Numerical T (RK4) = crossing not reached within max_t")
    else:
        print("Numerical T (RK4) = {:.6f}".format(T_numerical))
        print("Difference = {:.6e}".format(T_analytic - T_numerical))

if __name__ == "__main__":
    main()
