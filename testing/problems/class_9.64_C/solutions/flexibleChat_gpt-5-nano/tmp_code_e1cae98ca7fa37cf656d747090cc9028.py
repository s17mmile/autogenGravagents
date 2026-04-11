import math

# Physical constants and geometry
R_E = 6.371e6
mu = 3.986e14           # GM of Earth in m^3/s^2
g0 = 9.81                 # baseline gravity (m/s^2) for calibration
R_Earth = R_E

# Rocket data
m0 = 1.0e5
fuel_frac = 0.90
mf = m0 * (1 - fuel_frac)  # dry mass after burn (1e4 kg)
dotm = 900.0
tb = 100.0
F = dotm * 4000.0           # thrust in N (Ve * mdot)
A = math.pi * (0.2**2)       # cross-sectional area with radius 0.2 m
Cd = 0.2

# Drag model: D = k * v * |v|, where k = 0.5 * rho * Cd * A

def drag_k(rho):
    return 0.5 * rho * Cd * A

# Gravity models

def g_const(_h):
    return g0

def g_var(_h):
    # g(r) = mu / (R_E + h)^2
    return mu / ((R_Earth + _h)**2)

# RK4 helper: derivative of state (h, v) with respect to time
# State: t, h, v. Burn flag selects mass variation and thrust applicability.

def derivatives(t, h, v, rho, burn, g_func):
    # Mass depending on burn phase with clamping to mf to avoid dipping below dry mass
    m = m0 - dotm * t
    if burn and m < mf:
        m = mf
    # If coast, mass is fixed at mf and thrust is zero
    thrust = F if burn else 0.0
    g = g_func(h)
    k = drag_k(rho)
    # Drag opposes velocity: Fd = -k * v * |v|
    drag = - k * v * abs(v)
    a = (thrust + drag - m * g) / m
    return a


def rk4_step(t, h, v, dt, rho, burn, g_func):
    a1 = derivatives(t, h, v, rho, burn, g_func)
    h1 = v
    v1 = a1

    a2 = derivatives(t + dt/2.0, h + dt/2.0 * h1, v + dt/2.0 * v1, rho, burn, g_func)
    h2 = v + dt/2.0 * v1
    v2 = a2

    a3 = derivatives(t + dt/2.0, h + dt/2.0 * h2, v + dt/2.0 * v2, rho, burn, g_func)
    h3 = v + dt/2.0 * v2
    v3 = a3

    a4 = derivatives(t + dt, h + dt * h3, v + dt * v3, rho, burn, g_func)
    h4 = v + dt * v3
    v4 = a4

    h_next = h + (dt/6.0) * (h1 + 2*h2 + 2*h3 + h4)
    v_next = v + (dt/6.0) * (v1 + 2*v2 + 2*v3 + v4)
    t_next = t + dt
    return t_next, h_next, v_next


def apex_interpolate(t_prev, h_prev, v_prev, t_next, h_next, v_next, dt):
    # Local linear interpolation for zero crossing, with optional refinement via a tiny quadratic step
    if v_next == v_prev:
        return h_prev
    t_zero = t_prev + (0.0 - v_prev) * dt / (v_next - v_prev)
    h_zero = h_prev + v_prev * (t_zero - t_prev)
    return h_zero


def burn_phase(rho, dt, g_func):
    t = 0.0
    h = 0.0
    v = 0.0
    while t < tb:
        step = min(dt, tb - t)
        t_prev, h_prev, v_prev = t, h, v
        t, h, v = rk4_step(t, h, v, step, rho, burn=True, g_func=g_func)
    return t, h, v


def coast_phase(t_burn, h_burn, v_burn, rho, dt, g_func):
    t = t_burn
    h = h_burn
    v = v_burn
    t_prev, h_prev, v_prev = t, h, v
    while True:
        t_next, h_next, v_next = rk4_step(t, h, v, dt, rho, burn=False, g_func=g_func)
        if v_next <= 0.0:
            h_apex = apex_interpolate(t_prev, h_prev, v_prev, t_next, h_next, v_next, dt)
            return h_apex
        t_prev, h_prev, v_prev = t, h, v
        t, h, v = t_next, h_next, v_next
        # Safety: cap coast time to avoid infinite loops in pathological cases
        if t - tb > 20000:
            return h


def baseline_height(rho, dt=0.05):
    t_burn, h_burn, v_burn = burn_phase(rho, dt, g_const)
    h_apex = coast_phase(t_burn, h_burn, v_burn, rho, dt, g_const)
    return h_apex


def variable_height(rho, dt=0.05):
    t_burn, h_burn, v_burn = burn_phase(rho, dt, g_var)
    h_apex = coast_phase(t_burn, h_burn, v_burn, rho, dt, g_var)
    return h_apex


def calibrate_rho(target_height=890000.0, dt=0.05, tol=1000.0, max_iter=60):
    rho_lo = 0.0
    h_lo = baseline_height(rho_lo, dt)
    rho_hi = 2.0
    h_hi = baseline_height(rho_hi, dt)
    iter_ext = 0
    # Extend bracket if needed
    while (h_lo - target_height) * (h_hi - target_height) > 0 and iter_ext < 20:
        rho_hi *= 2.0
        h_hi = baseline_height(rho_hi, dt)
        iter_ext += 1
        if rho_hi > 1e3:
            break
    if (h_lo - target_height) * (h_hi - target_height) > 0:
        raise RuntimeError("Bracket not found for rho calibration; adjust dt or target.")
    rho_mid = 0.5 * (rho_lo + rho_hi)
    h_mid = baseline_height(rho_mid, dt)
    for _ in range(max_iter):
        if abs(h_mid - target_height) <= tol:
            return rho_mid
        if h_mid > target_height:
            rho_lo, h_lo = rho_mid, h_mid
        else:
            rho_hi, h_hi = rho_mid, h_mid
        rho_mid = 0.5 * (rho_lo + rho_hi)
        h_mid = baseline_height(rho_mid, dt)
    return rho_mid


def main():
    rho_cal = calibrate_rho(target_height=890000.0, dt=0.05, tol=1000.0, max_iter=60)
    h_base = baseline_height(rho_cal, dt=0.05)
    h_var = variable_height(rho_cal, dt=0.05)
    print("Calibrated rho (kg/m^3):", rho_cal)
    print("Baseline apex height (m):", h_base)
    print("Maximum height with variable gravity (m):", h_var)

if __name__ == "__main__":
    main()
