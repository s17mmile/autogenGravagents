import math
import csv

# Global constants (SI units)
g0 = 9.81            # m/s^2 at sea level
R_e = 6371000.0      # Earth's radius in meters
m0 = 1.0e5           # initial mass (kg)
mf = 1.0e4           # final mass after burn (kg)
dmdot = 900.0        # mass flow rate (kg/s)
ve = 4000.0           # exhaust velocity (m/s)
T = dmdot * ve         # thrust during burn (N)
Cd = 0.2
r = 0.20               # rocket radius (m)
A = math.pi * r * r    # cross-sectional area (m^2)

# Density model factories

def density_log_linear(a, b):
    # rho(h) = 10^(-a*h_km + b), h_km = h/1000
    def rho(h_m):
        h_km = h_m / 1000.0
        return 10.0 ** (-a * h_km + b)
    return rho

def density_exp_scale(H_km, rho0=1.29):
    # rho(h) = rho0 * exp(-h/H), with H in meters
    H = H_km * 1000.0
    def rho(h_m):
        return rho0 * math.exp(-h_m / H)
    return rho

# Gravity model

def gravity(h_m):
    return g0 * (R_e / (R_e + h_m)) ** 2

# RK4 step helper

def rk4_step(state, t, dt, derivs):
    k1 = derivs(state, t)
    s2 = [state[i] + 0.5 * dt * k1[i] for i in range(3)]
    k2 = derivs(s2, t + 0.5 * dt)
    s3 = [state[i] + 0.5 * dt * k2[i] for i in range(3)]
    k3 = derivs(s3, t + 0.5 * dt)
    s4 = [state[i] + dt * k3[i] for i in range(3)]
    k4 = derivs(s4, t + dt)
    next_state = [state[i] + (dt/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) for i in range(3)]
    return next_state

# Build derivs from a density function

def make_derivs(density_func, mf_local):
    def derivs(state, t):
        h, v, m = state
        burn_active = m > mf_local
        T_now = T if burn_active else 0.0
        dm_dt = -dmdot if burn_active else 0.0
        g = gravity(h)
        rho = density_func(h)
        F_drag = -0.5 * rho * Cd * A * v * abs(v)
        a = (T_now + F_drag - m * g) / m
        return [v, a, dm_dt]
    return derivs

# Apex refinement within a RK4 step to locate v = 0 more accurately

def refine_apex(state, t, dt, derivs, sub_steps=8):
    h0, v0, m0 = state
    s = state[:]
    t0 = t
    subdt = dt / float(sub_steps)
    h_prev, v_prev = h0, v0
    for i in range(sub_steps):
        s = rk4_step(s, t0, subdt, derivs)
        h_now, v_now = s[0], s[1]
        if v_prev > 0 and v_now <= 0:
            denom = (v_prev - v_now)
            if denom != 0:
                ratio = v_prev / denom
                apex_h = h_prev + ratio * (h_now - h_prev)
                apex_t = t0 + (i * subdt) + ratio * subdt
            else:
                apex_h = h_now
                apex_t = t0 + (i * subdt) + subdt
            return apex_h, apex_t
        h_prev, v_prev = h_now, v_now
        t0 += subdt
    return None, None

# Single-case solver for a given density function and density params

def run_case(a, b, density_factory, dt=0.05, max_time=4000.0):
    density_func = density_factory(a, b)
    derivs = make_derivs(density_func, mf)

    state = [0.0, 0.0, m0]
    t = 0.0

    Hmax = 0.0
    apex_height = None
    apex_time = None
    burnout_h = None
    burnout_v = None
    apex_found = False

    while t < max_time:
        next_state = rk4_step(state, t, dt, derivs)
        t_next = t + dt
        h_next, v_next, m_next = next_state

        if h_next > Hmax:
            Hmax = h_next

        if burnout_h is None and m_next <= mf:
            burnout_h = h_next
            burnout_v = v_next

        if not apex_found:
            v_prev = state[1]
            if v_prev > 0 and v_next <= 0:
                apex_h, apex_t = refine_apex(state, t, dt, derivs, sub_steps=8)
                if apex_h is not None:
                    apex_height = apex_h
                    apex_time = apex_t
                    apex_found = True
                    if apex_height > Hmax:
                        Hmax = apex_height
        state = next_state
        t = t_next
        if apex_found and (burnout_h is not None):
            break

    if apex_height is None:
        apex_height = Hmax
        apex_time = t

    return {
        'Hmax_m': Hmax,
        'burnout_h_m': burnout_h,
        'burnout_v_ms': burnout_v,
        'apex_h_m': apex_height,
        'apex_time_s': apex_time
    }

# Sensitivity harness: grid over a_values and b_values, write CSV

def run_sensitivity_and_write_csv():
    a_values = [0.04, 0.05, 0.06]
    b_values = [0.10, 0.11, 0.12]
    filename = 'rho_sensitivity_results.csv'
    headers = ['scenario_id', 'a', 'b', 'Hmax_km', 'burnout_km', 'apex_km', 'apex_time_s', 'burnout_v_ms']
    rows = []
    scenario_id = 1
    for a in a_values:
        for b in b_values:
            res = run_case(a, b, density_log_linear, dt=0.05, max_time=4000.0)
            Hmax_km = res['Hmax_m'] / 1000.0
            burnout_km = (res['burnout_h_m'] / 1000.0) if res['burnout_h_m'] is not None else None
            apex_km = res['apex_h_m'] / 1000.0
            rows.append([scenario_id, a, b, Hmax_km, burnout_km, apex_km, res['apex_time_s'], res['burnout_v_ms']])
            scenario_id += 1

    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    print("Sensitivity CSV written to", filename)

# Baseline quick-run helper (log-linear baseline)

def run_baseline_and_print():
    a, b = 0.05, 0.11
    res = run_case(a, b, density_log_linear, dt=0.05, max_time=4000.0)
    print("Baseline Hmax_km:", res['Hmax_m']/1000.0)
    print("Baseline burnout_km:", (res['burnout_h_m']/1000.0) if res['burnout_h_m'] is not None else None)
    print("Baseline apex_time_s:", res['apex_time_s'])
    print("Baseline apex_km:", res['apex_h_m']/1000.0)

# Simple, optional constant-density baseline function for bounds

def density_constant(_h):
    rho0 = 1.29
    return rho0

def run_with_constant_density():
    def rho_const(h):
        return 1.29
    derivs = make_derivs(rho_const, mf)
    state = [0.0, 0.0, m0]
    t = 0.0
    dt = 0.05
    Hmax = 0.0
    apex_h = None
    apex_t = None
    burnout_h = None
    burnout_v = None
    while t < 4000:
        ns = rk4_step(state, t, dt, derivs)
        h,nv,m = ns[0], ns[1], ns[2]
        if h > Hmax:
            Hmax = h
        if burnout_h is None and m <= mf:
            burnout_h = h
            burnout_v = nv
        if nv > 0 and ns[1] <= 0:
            apex_h = h
            apex_t = t
            break
        state = ns
        t += dt
    print("Constant-density baseline Hmax_m:", Hmax)
    print("apex_h_est:", apex_h, "apex_t_s:", apex_t)

# Driver

def main():
    print("Running baseline quick-run for a=0.05, b=0.11...")
    run_baseline_and_print()
    print("Running sensitivity study and writing CSV...")
    run_sensitivity_and_write_csv()
    print("Optional constant-density baseline for bounding drag:\n")
    run_with_constant_density()

if __name__ == '__main__':
    main()
