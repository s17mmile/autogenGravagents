import math
from dataclasses import dataclass


@dataclass
class Params:
    m0: float          # initial total mass (kg)
    fuel_fraction: float  # fraction of mass that is fuel
    tb: float          # burn time (s)
    ve: float          # exhaust velocity (m/s)
    g: float           # gravitational acceleration (m/s^2)
    rho: float         # air density (kg/m^3)
    Cd: float          # drag coefficient (dimensionless)
    r: float           # rocket radius (m)


def analytic_burn_outputs(m0, m_final, tb, ve, g):
    # Validate inputs to avoid invalid logs or divisions
    if not (m_final > 0 and m0 > m_final and tb > 0 and ve > 0):
        raise ValueError("Invalid mass/time parameters for analytic burn")
    ln_ratio = math.log(m0 / m_final)
    v_b = ve * ln_ratio - g * tb
    mdot = (m0 - m_final) / tb
    h_burn = (ve / mdot) * (m0 - m_final * (ln_ratio + 1.0)) - 0.5 * g * tb * tb
    return v_b, h_burn


def post_burn_delta_h(v_b, g, m_final, rho, Cd, r):
    A = math.pi * r * r
    k = 0.5 * rho * Cd * A
    alpha = k / m_final
    delta_h = (1.0 / (2.0 * alpha)) * math.log((g + alpha * v_b * v_b) / g)
    return delta_h, alpha


def burn_with_drag_numerical(m0, m_final, tb, ve, g, rho, Cd, r, steps=4000):
    # RK4 integration for burn phase with drag during burn
    mdot = (m0 - m_final) / tb
    A = math.pi * r * r
    k = 0.5 * rho * Cd * A

    v = 0.0
    m = m0
    h = 0.0

    def f(state):
        vv, mm, hh = state
        dvdt = (mdot * ve) / mm - g - (k / mm) * (vv * vv)
        dmdt = -mdot
        dhdt = vv
        return dvdt, dmdt, dhdt

    state = (v, m, h)
    dt = tb / steps
    for _ in range(steps):
        k1 = f(state)
        state2 = (state[0] + k1[0] * dt/2.0, state[1] + k1[1] * dt/2.0, state[2] + k1[2] * dt/2.0)
        k2 = f(state2)
        state3 = (state[0] + k2[0] * dt/2.0, state[1] + k2[1] * dt/2.0, state[2] + k2[2] * dt/2.0)
        k3 = f(state3)
        state4 = (state[0] + k3[0] * dt, state[1] + k3[1] * dt, state[2] + k3[2] * dt)
        k4 = f(state4)
        v = state[0] + (dt/6.0) * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
        m = state[1] + (dt/6.0) * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
        h = state[2] + (dt/6.0) * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])
        state = (v, m, h)
        if m <= m_final:
            break
    return v, h


def compute_H_max(params, include_drag_during_burn=False, burn_steps=4000):
    m0 = params.m0
    mfuel = m0 * params.fuel_fraction
    m_final = m0 - mfuel
    tb = params.tb
    ve = params.ve
    g = params.g
    rho = params.rho
    Cd = params.Cd
    r = params.r

    if not (m_final > 0 and m0 > m_final and tb > 0):
        raise ValueError("Invalid mass or burn time configuration")

    # analytic burn path
    v_b, h_burn = analytic_burn_outputs(m0, m_final, tb, ve, g)
    delta_h, alpha = post_burn_delta_h(v_b, g, m_final, rho, Cd, r)
    H_max = h_burn + delta_h

    if include_drag_during_burn:
        v_b_num, h_burn_num = burn_with_drag_numerical(m0, m_final, tb, ve, g, rho, Cd, r, steps=burn_steps)
        delta_h_num, _ = post_burn_delta_h(v_b_num, g, m_final, rho, Cd, r)
        H_max_num = h_burn_num + delta_h_num
        return {
            "H_max": H_max,
            "v_burn": v_b, "h_burn": h_burn, "delta_h_drag": delta_h,
            "v_burn_num": v_b_num, "h_burn_num": h_burn_num, "delta_h_drag_num": delta_h_num,
            "H_max_num": H_max_num,
        }
    else:
        return {
            "H_max": H_max,
            "v_burn": v_b, "h_burn": h_burn, "delta_h_drag": delta_h,
        }


def main():
    # Problem data from the prompt
    m0 = 1e5
    fuel_fraction = 0.9
    tb = 100.0
    ve = 4000.0
    g = 9.81
    rho = 1.225
    Cd = 0.2
    r = 0.20

    params = Params(m0=m0, fuel_fraction=fuel_fraction, tb=tb, ve=ve, g=g, rho=rho, Cd=Cd, r=r)

    res_no_drag = compute_H_max(params, include_drag_during_burn=False, burn_steps=4000)
    res_with_drag = compute_H_max(params, include_drag_during_burn=True, burn_steps=4000)

    print("No burn drag results:")
    print("  v_b = {:.6f} m/s".format(res_no_drag["v_burn"]))
    print("  h_burn = {:.6f} m".format(res_no_drag["h_burn"]))
    print("  delta_h_drag = {:.6f} m".format(res_no_drag["delta_h_drag"]))
    print("  H_max = {:.6f} m".format(res_no_drag["H_max"]))
    print()
    print("Burn-drag results (numerical):")
    print("  v_b = {:.6f} m/s".format(res_with_drag["v_burn_num"]))
    print("  h_burn = {:.6f} m".format(res_with_drag["h_burn_num"]))
    print("  delta_h_drag = {:.6f} m".format(res_with_drag["delta_h_drag_num"]))
    print("  H_max = {:.6f} m".format(res_with_drag["H_max_num"]))


if __name__ == "__main__":
    main()
