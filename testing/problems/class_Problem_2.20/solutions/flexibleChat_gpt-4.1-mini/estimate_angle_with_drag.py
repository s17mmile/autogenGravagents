# filename: estimate_angle_with_drag.py
import math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# Constants
mass = 10.0  # kg
muzzle_velocity = 140.0  # m/s
g = 9.81  # m/s^2
rho = 1.225  # kg/m^3 (air density at sea level)
Cd = 0.47  # drag coefficient (approximate for a sphere)
diameter = 0.1  # m (assumed projectile diameter)
A = math.pi * (diameter / 2) ** 2  # cross-sectional area

# Equations of motion with drag
# state vector: [x, y, vx, vy]
def equations(t, state):
    x, y, vx, vy = state
    v = math.sqrt(vx**2 + vy**2)
    drag_force = 0.5 * Cd * rho * A * v**2
    drag_acceleration = drag_force / mass
    ax = -drag_acceleration * (vx / v) if v != 0 else 0
    ay = -g - drag_acceleration * (vy / v) if v != 0 else -g
    return [vx, vy, ax, ay]

# Function to simulate projectile motion and return horizontal range for a given launch angle
def simulate_range(angle_deg):
    angle_rad = math.radians(angle_deg)
    vx0 = muzzle_velocity * math.cos(angle_rad)
    vy0 = muzzle_velocity * math.sin(angle_rad)
    state0 = [0, 0, vx0, vy0]

    def hit_ground(t, y):
        return y[1]  # y position
    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(equations, [0, 1000], state0, events=hit_ground, max_step=0.1)
    x_impact = sol.y[0, -1]
    return x_impact

# Objective function for minimization: squared difference from target range
target_range = 1000.0

def objective(angle_deg):
    range_achieved = simulate_range(angle_deg)
    return (range_achieved - target_range) ** 2

# Search for angle between 0 and 90 degrees
result = minimize_scalar(objective, bounds=(0, 90), method='bounded', options={'xatol': 0.01})

angle_with_drag = result.x
range_with_drag = simulate_range(angle_with_drag)

result_message = (f"Estimated angle with air resistance to hit {target_range} m target: {angle_with_drag:.2f} degrees\n"
                  f"Range achieved at this angle: {range_with_drag:.2f} m")

result_message
