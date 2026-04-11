import math

# Physical constants and ball parameters (as given)
c_W = 0.5
radius = 0.05  # meters
mass = 0.20    # kg
rho = 1.225      # kg/m^3 (air density at sea level)
A = math.pi * radius * radius
g = 9.81          # m/s^2

class BallProjectile:
    def __init__(self, c_W: float = 0.5, radius: float = 0.05, mass: float = 0.20, rho: float = 1.225, g: float = 9.81):
        self.c_W = c_W
        self.radius = radius
        self.mass = mass
        self.rho = rho
        self.A = math.pi * radius * radius
        self.g = g
        # Quadratic drag parameter k in a_x = -k * v * v_x, a_y = -g - k * v * v_y
        self.k = 0.5 * rho * c_W * self.A / mass

    def derivatives(self, state):
        x, y, vx, vy = state
        v = math.hypot(vx, vy)
        ax = -self.k * v * vx
        ay = -self.g - self.k * v * vy
        return vx, vy, ax, ay

    def rk4_step(self, state, dt: float):
        x1, y1, vx1, vy1 = state
        dx1, dy1, dvx1, dvy1 = self.derivatives(state)

        state2 = [x1 + dx1 * dt/2.0, y1 + dy1 * dt/2.0, vx1 + dvx1 * dt/2.0, vy1 + dvy1 * dt/2.0]
        dx2, dy2, dvx2, dvy2 = self.derivatives(state2)

        state3 = [x1 + dx2 * dt/2.0, y1 + dy2 * dt/2.0, vx1 + dvx2 * dt/2.0, vy1 + dvy2 * dt/2.0]
        dx3, dy3, dvx3, dvy3 = self.derivatives(state3)

        state4 = [x1 + dx3 * dt, y1 + dy3 * dt, vx1 + dvx3 * dt, vy1 + dvy3 * dt]
        dx4, dy4, dvx4, dvy4 = self.derivatives(state4)

        x = x1 + dt * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6.0
        y = y1 + dt * (dy1 + 2*dy2 + 2*dy3 + dy4) / 6.0
        vx = vx1 + dt * (dvx1 + 2*dvx2 + 2*dvx3 + dvx4) / 6.0
        vy = vy1 + dt * (dvy1 + 2*dvy2 + 2*dvy3 + dvy4) / 6.0
        return [x, y, vx, vy]

    def simulate_to_xf(self, v0: float, theta_deg: float, x_f: float, h_f: float, dt: float = 0.001, tmax: float = 60.0, tol: float = 1e-3):
        theta = math.radians(theta_deg)
        state = [0.0, 0.0, v0 * math.cos(theta), v0 * math.sin(theta)]
        t = 0.0
        x_prev, y_prev = state[0], state[1]
        while t < tmax:
            x, y, vx, vy = state
            if y < 0.0 and t > 0.0:
                return None  # hit ground before fence
            if x >= x_f:
                dx = x - x_prev
                if dx == 0:
                    y_at_xf = y
                else:
                    frac = (x_f - x_prev) / dx
                    y_at_xf = y_prev + frac * (y - y_prev)
                return y_at_xf
            state = self.rk4_step(state, dt)
            t += dt
            x_prev, y_prev = x, y
        return None


def min_v0_for_theta(theta_deg: float, x_f: float, h_f: float, v_min: float, v_max: float, tol: float, projectile: BallProjectile):
    y_min = projectile.simulate_to_xf(v_min, theta_deg, x_f, h_f, dt=0.001, tmax=60.0, tol=tol)
    y_max = projectile.simulate_to_xf(v_max, theta_deg, x_f, h_f, dt=0.001, tmax=60.0, tol=tol)

    if y_max is None or y_max < h_f - tol:
        return (None, None)
    if y_min is not None and y_min >= h_f - tol:
        return (v_min, y_min)

    left, right = v_min, v_max
    best_v = None
    best_y = None
    for _ in range(50):
        mid = 0.5 * (left + right)
        y_mid = projectile.simulate_to_xf(mid, theta_deg, x_f, h_f, dt=0.001, tmax=60.0, tol=tol)
        if y_mid is not None and y_mid >= h_f - tol:
            best_v = mid
            best_y = y_mid
            right = mid
        else:
            left = mid
        if right - left < 1e-4:
            break
    if best_v is None:
        return (None, None)
    return (best_v, best_y)


def find_min_v0(x_f: float, h_f: float, projectile: BallProjectile, v_min: float = 5.0, v_max: float = 70.0, tol: float = 1e-3, theta_start: float = 5.0, theta_end: float = 85.0, theta_step: float = 1.0):
    best = None
    best_theta = None
    best_y = None
    theta = theta_start
    while theta <= theta_end + 1e-9:
        v_min_theta, y_theta = min_v0_for_theta(theta, x_f, h_f, v_min, v_max, tol, projectile)
        if v_min_theta is not None:
            if best is None or v_min_theta < best:
                best = v_min_theta
                best_theta = theta
                best_y = y_theta
        theta += theta_step
    return {"min_v0": best, "optimal_theta": best_theta, "y_at_xf": best_y, "success": best is not None}

if __name__ == '__main__':
    # Example fence: distance x_f = 25 m, height h_f = 3.0 m
    X_F = 25.0
    H_F = 3.0
    projectile = BallProjectile()
    result = find_min_v0(X_F, H_F, projectile, v_min=5.0, v_max=70.0, tol=1e-3, theta_start=5.0, theta_end=85.0, theta_step=1.0)
    print(result)
