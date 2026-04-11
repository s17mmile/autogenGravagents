import math
from typing import List, Tuple, Optional

# Derivatives for the state [x, y, vx, vy] under quadratic drag
# v = sqrt(vx^2 + vy^2); a_drag = (1/2) * rho * Cd * A * v^2 / m
# Acceleration components opposite to velocity

def derivatives(state: List[float], rho: float, Cd: float, A: float, m: float, g: float = 9.81) -> List[float]:
    x, y, vx, vy = state
    v = math.hypot(vx, vy)
    if v > 0.0:
        ax = -0.5 * rho * Cd * A * v * vx / m
        ay = -g - 0.5 * rho * Cd * A * v * vy / m
    else:
        ax = 0.0
        ay = -g
    return [vx, vy, ax, ay]


def rk4_step(state: List[float], rho: float, Cd: float, A: float, m: float, g: float = 9.81, dt: float = 1e-3) -> List[float]:
    k1 = derivatives(state, rho, Cd, A, m, g)
    s2 = [state[i] + 0.5 * dt * k1[i] for i in range(4)]
    k2 = derivatives(s2, rho, Cd, A, m, g)
    s3 = [state[i] + 0.5 * dt * k2[i] for i in range(4)]
    k3 = derivatives(s3, rho, Cd, A, m, g)
    s4 = [state[i] + dt * k3[i] for i in range(4)]
    k4 = derivatives(s4, rho, Cd, A, m, g)
    next_state = [state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(4)]
    return next_state


def range_with_drag_rk4(v0: float, m: float, Cd: float, A: float, rho: float, theta_deg: float, g: float = 9.81, dt: float = 1e-3, max_steps: int = 600000) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Returns (R, t_flight, h_peak) or (None, None, None) if not landed within steps
    theta = math.radians(theta_deg)
    state = [0.0, 0.0, v0 * math.cos(theta), v0 * math.sin(theta)]  # x, y, vx, vy
    t = 0.0
    x_prev, y_prev = 0.0, 0.0
    max_y = 0.0

    for _ in range(int(max_steps)):
        x, y, vx, vy = state
        x_prev, y_prev = x, y
        state = rk4_step(state, rho, Cd, A, m, g, dt)
        t += dt
        x, y, vx, vy = state
        if y > max_y:
            max_y = y
        if y < 0.0:
            # Interpolate to estimate R at y = 0 crossing
            if y_prev != y:
                f = y_prev / (y_prev - y)
                R = x_prev + f * (x - x_prev)
            else:
                R = x
            return R, t, max_y
    return None, None, max_y  # did not land within time budget


def solve_for_angles(
    v0: float,
    m: float,
    Cd: float,
    A: float,
    rho: float,
    R_target: float,
    theta_bounds: Tuple[float, float] = (0.0, 90.0),
    coarse_step: float = 1.0,
    tol_R: float = 1.0,
    g: float = 9.81,
    dt: float = 1e-3,
    max_steps: int = 600000
) -> List[Tuple[float, float, float, float]]:
    # Returns a list of tuples: (theta_deg, R, t_flight, h_peak)
    low, high = theta_bounds
    thetas = []
    th = low
    while th <= high:
        thetas.append(th)
        th += coarse_step
    # Compute R for each theta
    samples = []  # (theta, R, t, h)
    for th in thetas:
        R, t_f, h = range_with_drag_rk4(v0, m, Cd, A, rho, th, g=g, dt=dt, max_steps=max_steps)
        samples.append((th, R, t_f, h))

    # Bracket roots where R crosses R_target
    results = []
    for i in range(len(samples) - 1):
        th1, R1, t1, h1 = samples[i]
        th2, R2, t2, h2 = samples[i+1]
        if R1 is None or R2 is None:
            continue
        f1 = R1 - R_target
        f2 = R2 - R_target
        if f1 == 0.0:
            results.append((th1, R1, t1, h1))
        if f1 * f2 <= 0.0:
            # Root exists in [th1, th2], refine with bisection on theta
            a, b = th1, th2
            Ra, Rb = R1, R2
            ta, tb = t1, t2
            ha, hb = h1, h2
            for _ in range(40):
                mid = 0.5 * (a + b)
                Rm, tm, hm = range_with_drag_rk4(v0, m, Cd, A, rho, mid, g=g, dt=dt, max_steps=max_steps)
                if Rm is None:
                    # If not landed, adjust bracket by reducing step size
                    a = mid
                    continue
                fm = Rm - R_target
                if abs(fm) <= tol_R:
                    results.append((mid, Rm, tm, hm))
                    break
                # Decide which subinterval to keep
                if f1 * fm <= 0:
                    b, Rb = mid, Rm
                    tb, hb = tm, hm
                    fb = fm
                else:
                    a, Ra = mid, Rm
                    ta, ha = tm, hm
                    f1 = fm
            else:
                # If not converged, record best effort at midpoint
                mid = 0.5 * (a + b)
                Rm, tm, hm = range_with_drag_rk4(v0, m, Cd, A, rho, mid, g=g, dt=dt, max_steps=max_steps)
                if Rm is not None:
                    results.append((mid, Rm, tm, hm))

    # Return as list of tuples (theta, R, t, h)
    return results

# Demonstration with illustrative parameters (not tied to Figure 2-3)
if __name__ == "__main__":
    # Illustrative drag parameters (example only)
    v0 = 140.0
    m = 10.0
    d = 0.05
    A = math.pi * (d/2.0)**2
    Cd = 0.30
    rho = 1.225
    R_target = 1000.0

    sols = solve_for_angles(v0=v0, m=m, Cd=Cd, A=A, rho=rho, R_target=R_target,
                            theta_bounds=(0.0, 90.0), coarse_step=1.0, tol_R=1.0,
                            g=9.81, dt=1e-3, max_steps=600000)

    print("Drag-based angle solutions (illustrative parameters):")
    if not sols:
        print("No solution found within the scanned range and tolerance.")
    else:
        for th, R, t_f, h in sols:
            print("theta = {:6.3f} deg, R = {:8.3f} m, t = {:8.4f} s, h_peak = {:8.4f} m".format(th, R, t_f, h))
