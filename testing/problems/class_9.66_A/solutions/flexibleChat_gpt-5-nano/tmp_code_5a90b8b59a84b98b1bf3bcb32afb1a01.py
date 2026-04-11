import math

# Preset physical constants and geometry
m0 = 0.054            # initial total mass (kg)
delta_m = 0.011        # propellant mass burned (kg)
t_burn = 1.5            # burn time (s)
mdot = delta_m / t_burn  # mass flow rate (kg/s)
rho = 1.225             # air density (kg/m^3)
Cd = 0.75               # drag coefficient (dimensionless)
d = 0.024               # rocket diameter (m)
A = math.pi * (d / 2.0) ** 2
k = 0.5 * rho * Cd * A    # drag coefficient in D = k v^2
u_e = 800.0              # exhaust speed (m/s)
I_t = 8.5                # total impulse (N*s)

# Thrust definitions
F_dot = mdot * u_e        # thrust estimate from mass flow and exhaust speed
F_avg = I_t / t_burn       # thrust if distributed evenly over burn time

# End mass sanity
m_end = m0 - mdot * t_burn
if m_end <= 0:
    m_end = 1e-12

# Analytic closed-form burnout speed for a given thrust F

def analytic_v(F):
    m_end_local = m_end
    # ln(m0/m_end) appears in the closed-form, guard against zero/negative
    if m_end_local <= 0:
        m_end_local = 1e-12
    return math.sqrt(F / k) * math.tanh((math.sqrt(F * k) / mdot) * math.log(m0 / m_end_local))

# RK4 integration for validation: dv/dt = (F - k v^2) / m(t), dm/dt = -mdot

def rk4_step(v, m, F, dt):
    def f(v_, m_):
        dvdt = (F - k * v_ * v_) / m_
        dmdt = -mdot
        return dvdt, dmdt
    dv1, dm1 = f(v, m)
    dv2, dm2 = f(v + 0.5 * dt * dv1, m + 0.5 * dt * dm1)
    dv3, dm3 = f(v + 0.5 * dt * dv2, m + 0.5 * dt * dm2)
    dv4, dm4 = f(v + dt * dv3, m + dt * dm3)
    v_next = v + (dt / 6.0) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
    m_next = m + (dt / 6.0) * (dm1 + 2 * dm2 + 2 * dm3 + dm4)
    return v_next, m_next


def rk4_integrate(F, steps=20000):
    v = 0.0
    m = m0
    t = 0.0
    dt = t_burn / steps
    for i in range(steps):
        if t >= t_burn:
            break
        if t + dt > t_burn:
            dt = t_burn - t
        v, m = rk4_step(v, m, F, dt)
        t += dt
    return v, m, t

# Compute burnout speeds for both thrust definitions
v_analytic_F = analytic_v(F_dot)
v_analytic_Favg = analytic_v(F_avg)

v_rk4_F, m_final_F, t_final_F = rk4_integrate(F_dot, steps=20000)
v_rk4_Favg, m_final_Favg, t_final_Favg = rk4_integrate(F_avg, steps=20000)

# Output results
print("Analytic burnout speed (F = m_dot * u_e):", v_analytic_F)
print("Analytic burnout speed (F = I_t / t_burn):", v_analytic_Favg)
print("RK4 burnout speed (F = m_dot * u_e):", v_rk4_F)
print("RK4 burnout speed (F = I_t / t_burn):", v_rk4_Favg)
