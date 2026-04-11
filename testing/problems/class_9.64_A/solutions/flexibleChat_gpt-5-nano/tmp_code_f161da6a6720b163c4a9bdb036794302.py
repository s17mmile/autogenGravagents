import math

# Given constants
m0 = 1.0e5          # initial total mass (kg)
dry_fraction = 0.1  # 10 percent of mass is dry
mdry = m0 * dry_fraction
mf = m0 - mdry      # fuel mass

tburn = 100.0        # burn time (s)
ve = 4000.0          # exhaust velocity (m/s)
g = 9.81             # gravity (m/s^2)

mdot = mf / tburn     # mass flow rate (kg/s)

# Analytic burnout velocity for cross-check
v_burn_analytic = ve * math.log(m0 / mdry) - g * tburn

# Numerical integration to estimate burnout altitude with explicit Euler for y
# Use a step that exactly reaches tburn

dt = 0.01
t = 0.0
v = 0.0
y = 0.0
while t < tburn:
    dt_step = min(dt, tburn - t)
    m = m0 - mdot * t
    a = (mdot * ve) / m - g
    v_old = v
    v = v + a * dt_step
    y = y + v_old * dt_step  # explicit Euler update for y
    t += dt_step

# Burnout quantities
y_burn = y
v_burn_num = v

# Coast phase (no thrust)
h_coast = (v_burn_num ** 2) / (2.0 * g)
y_max = y_burn + h_coast

# Simple cross-check assertion with a tolerance
tol = 1.0  # m/s tolerance for velocity comparison
if abs(v_burn_num - v_burn_analytic) > tol:
    print("Warning: burnout velocity numerical vs analytic differ by more than the tolerance.")

print("Analytic burnout velocity (m/s):", v_burn_analytic)
print("Numeric burnout velocity (m/s):", v_burn_num)
print("Burnout altitude (m):", y_burn)
print("Coast height (m):", h_coast)
print("Maximum altitude (m):", y_max)
