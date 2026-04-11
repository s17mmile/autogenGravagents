import math

# Physical constants
R_E = 6.371e6      # Earth radius (m)
RHO = 1.225        # Air density at sea level (kg/m^3)
g = 9.81             # Gravity (m/s^2)
FALK_LAT_S = 51.7    # Falkland latitude (deg, South)
REF_LAT_S = 50.0     # Firing latitude (deg, South) for baseline reference

# No-drag baseline helper

def no_drag_results(v0, theta_deg, ref_lat_S=REF_LAT_S):
    theta = math.radians(theta_deg)
    R = (v0*v0) * math.sin(2.0*theta) / g
    delta_deg = (R / R_E) * (180.0 / math.pi)  # southward latitude increase
    land_lat_S = ref_lat_S + delta_deg
    miss_km = abs(FALK_LAT_S - land_lat_S) * 111.32
    direction = 'north' if land_lat_S < FALK_LAT_S else 'south'
    return {
        'landed': True,
        'range_m': R,
        'landing_lat_S_deg': land_lat_S,
        'miss': {'direction': direction, 'km': miss_km}
    }

# Drag-inclusive trajectory using RK4 (2D: x = downrange south, y = height)

def run_drag_case(m, D, Cd, v0, theta_deg, dt=0.01, max_time=600.0):
    theta = math.radians(theta_deg)
    A = math.pi * (D*D) / 4.0
    B = (0.5 * RHO * Cd * A) / m  # quadratic drag coefficient

    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)
    x = 0.0
    y = 0.0

    def deriv(state):
        x0, y0, vx0, vy0 = state
        v = math.hypot(vx0, vy0)
        ax = -B * v * vx0
        ay = -g - B * v * vy0
        return [vx0, vy0, ax, ay]

    t = 0.0
    state = [x, y, vx, vy]

    while True:
        k1 = deriv(state)
        s2 = [state[i] + 0.5*dt*k1[i] for i in range(4)]
        k2 = deriv(s2)
        s3 = [state[i] + 0.5*dt*k2[i] for i in range(4)]
        k3 = deriv(s3)
        s4 = [state[i] + dt*k3[i] for i in range(4)]
        k4 = deriv(s4)
        new = [state[i] + (dt/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) for i in range(4)]
        t += dt
        # Check landing (y crosses zero)
        if new[1] < 0.0:
            y0 = state[1]
            y1 = new[1]
            x0 = state[0]
            x1 = new[0]
            if y0 == y1:
                x_land = x1
            else:
                frac = y0 / (y0 - y1)
                x_land = x0 + (x1 - x0) * frac
            # Latitude mapping from downrange x_land
            delta_deg = (x_land / R_E) * (180.0 / math.pi)
            land_lat_S = REF_LAT_S + delta_deg
            miss_dir = 'north' if land_lat_S < FALK_LAT_S else 'south'
            miss_km = abs(FALK_LAT_S - land_lat_S) * 111.32
            return {
                'landed': True,
                'x_land_m': x_land,
                't_land_s': t,
                'landing_lat_S_deg': land_lat_S,
                'miss': {'direction': miss_dir, 'km': miss_km}
            }
        state = new
        if t > max_time:
            return {'landed': False, 'x_land_m': None, 't_land_s': None, 'landing_lat_S_deg': None, 'miss': {'direction': None, 'km': None}}

# Report generator (human-friendly)

def report_summary(results):
    parts = []
    nd = results.get('no_drag', {})
    parts.append("No-drag baseline: range {:.2f} km, landing latitude {:.3f} S, miss {:.1f} km {} of Falklands".format(
        nd.get('range_m', 0.0)/1000.0,
        nd.get('landing_lat_S_deg', 0.0),
        nd.get('miss', {}).get('km', 0.0),
        nd.get('miss', {}).get('direction', '')))
    for k in ['drag_case_A', 'drag_case_B']:
        dc = results.get(k, {})
        if dc:
            lat = dc.get('landing_lat_S_deg', None)
            miss = dc.get('miss', {})
            if lat is not None:
                parts.append("{}: landing latitude {:.3f} S, miss {:.1f} km {} of Falklands".format(
                    k, lat, miss.get('km', 0.0), miss.get('direction', '')))
            else:
                parts.append("{}: landing not reached within time window".format(k))
    return "; ".join(parts)


def main():
    v0 = 800.0
    theta_deg = 37.0

    # Subtask 1: No-drag baseline
    nd = no_drag_results(v0, theta_deg)

    # Subtask 3 & 4: Drag-inclusive cases (two representative shells)
    case_A = run_drag_case(m=600.0, D=0.15, Cd=0.30, v0=v0, theta_deg=theta_deg)
    case_B = run_drag_case(m=300.0, D=0.12, Cd=0.50, v0=v0, theta_deg=theta_deg)

    results = {
        'no_drag': nd,
        'drag_case_A': case_A,
        'drag_case_B': case_B
    }

    # Print a compact report sentence
    print("Summary: ", report_summary(results))

    return results

if __name__ == '__main__':
    main()
