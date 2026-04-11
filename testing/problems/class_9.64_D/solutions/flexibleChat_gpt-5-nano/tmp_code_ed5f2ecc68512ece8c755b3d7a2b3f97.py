import math

# Physical constants and geometry
g0 = 9.81  # m/s^2
R_e = 6371000.0  # Earth radius in meters
m0 = 1.0e5  # initial mass (kg)
mf = 1.0e4  # final mass after burn (kg)
dmdot = 900.0  # mass flow rate (kg/s)
ve = 4000.0  # exhaust velocity (m/s)
T = dmdot * ve  # thrust during burn (N)
Cd = 0.2
r = 0.20  # rocket radius (m)
A = math.pi * r * r  # cross-sectional area (m^2)

def gravity(h_m):
    # Gravitational acceleration as a function of altitude using g(h) = g0*(R/(R+h))^2
    return g0 * (R_e / (R_e + h_m))**2


def density(h_m):
    # Atmospheric density model: rho(h) = 10^(-0.05*h_km + 0.11), h in meters
    h_km = h_m / 1000.0
    return 10.0 ** (-0.05 * h_km + 0.11)

# Dynamics: state = [h, v, m]
# t in seconds, h in meters, v in m/s, m in kg

def derivatives(state, t):
    h, v, m = state
    burn_active = m > mf
    T_now = T if burn_active else 0.0
    dm_dt = -dmdot if burn_active else 0.0
    g = gravity(h)
    rho = density(h)
    # Drag opposite to velocity: F_drag = -0.5 * rho * Cd * A * v * |v|
    F_drag = -0.5 * rho * Cd * A * v * abs(v)
    a = (T_now + F_drag - m * g) / m
    return [v, a, dm_dt]


def rk4_step(state, t, dt):
    k1 = derivatives(state, t)
    state2 = [state[i] + 0.5 * dt * k1[i] for i in range(3)]
    k2 = derivatives(state2, t + 0.5 * dt)
    state3 = [state[i] + 0.5 * dt * k2[i] for i in range(3)]
    k3 = derivatives(state3, t + 0.5 * dt)
    state4 = [state[i] + dt * k3[i] for i in range(3)]
    k4 = derivatives(state4, t + dt)
    next_state = [state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(3)]
    return next_state


def refine_apex(state, t, dt, sub_steps=8):
    # Refine the apex (v = 0) within the interval [t, t+dt] using sub-steps
    h0, v0, m0 = state
    h_prev, v_prev = h0, v0
    s = state[:]
    tt = t
    subdt = dt / float(sub_steps)
    for i in range(sub_steps):
        s = rk4_step(s, tt, subdt)
        h_now, v_now = s[0], s[1]
        if v_prev > 0 and v_now <= 0:
            denom = (v_prev - v_now)
            if denom != 0:
                ratio = v_prev / denom
                apex_h = h_prev + ratio * (h_now - h_prev)
                apex_t = t + (i * subdt) + ratio * subdt
            else:
                apex_h = h_now
                apex_t = tt + subdt
            return apex_h, apex_t
        h_prev, v_prev = h_now, v_now
        tt += subdt
    return None, None


def main():
    dt = 0.05  # integration step in seconds

    # Initial state: height h (m), velocity v (m/s), mass m (kg)
    state = [0.0, 0.0, m0]
    t = 0.0

    Hmax = 0.0  # maximum height observed (m)
    burnout_h = None
    burnout_v = None
    apex_height = None
    apex_time = None
    apex_found = False

    max_time = 4000.0

    while t < max_time:
        next_state = rk4_step(state, t, dt)
        t_next = t + dt
        h_next, v_next, m_next = next_state

        # Update maximum height with the end-of-step height
        if h_next > Hmax:
            Hmax = h_next

        # Burnout detection (mass crosses mf)
        if burnout_h is None and m_next <= mf:
            burnout_h = h_next
            burnout_v = v_next

        # Apex detection: velocity crosses from positive to non-positive within this step
        if not apex_found:
            v_prev = state[1]
            if v_prev > 0 and v_next <= 0:
                apex_h, apex_t = refine_apex(state, t, dt, sub_steps=8)
                if apex_h is None:
                    # Fallback to linear interpolation if refinement fails
                    denom = (v_prev - v_next)
                    if denom != 0:
                        ratio = v_prev / denom
                        apex_h = state[0] + ratio * (h_next - state[0])
                        apex_t = t + ratio * dt
                    else:
                        apex_h = h_next
                        apex_t = t_next
                apex_height = apex_h
                apex_time = apex_t
                apex_found = True
                if apex_height > Hmax:
                    Hmax = apex_height

        # Prepare for next iteration
        state = next_state
        t = t_next

        if apex_found and burnout_h is not None:
            break

    # Fallbacks in case apex or burnout were not observed within max_time
    if apex_height is None:
        apex_height = Hmax
        apex_time = t

    # Output results
    print("Maximum altitude (m):", Hmax)
    print("Maximum altitude (km):", Hmax / 1000.0)
    if burnout_h is not None:
        print("Burnout altitude (m):", burnout_h)
        print("Burnout altitude (km):", burnout_h / 1000.0)
        print("Burnout velocity (m/s):", burnout_v)
    else:
        print("Burnout not reached within simulation time.")
    print("Apex height estimate (m):", apex_height)
    print("Apex height estimate (km):", apex_height / 1000.0)
    print("Apex occur time (s):", apex_time)

if __name__ == "__main__":
    main()
