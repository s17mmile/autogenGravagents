import math

# Given constants
m0 = 1.0e5          # initial total mass (kg)
dry_fraction = 0.1  # 10% is dry mass
mdry = m0 * dry_fraction
mf = m0 - mdry      # fuel mass
tburn = 100.0        # burn time (s)
ve = 4000.0          # exhaust velocity (m/s)
g = 9.81             # gravity (m/s^2)
ddot = mf / tburn     # mass flow rate (kg/s)

# Analytic burnout velocity
v_burn_analytic = ve * math.log(m0 / mdry) - g * tburn

# Numerical integration to estimate burnout altitude
dt = 0.01
nsteps = int(tburn / dt)
v = 0.0
y = 0.0
for i in range(nsteps):
    t = i * dt
    m = m0 - mdot * t
    a = (mdot * ve) / m - g
    v += a * dt
    y += v * dt

y_burn = y            # burnout altitude
v_burn_num = v        # burnout velocity from numeric integration

# Coast phase (no thrust): maximum height after burnout
h_coast = v_burn_num * v_burn_num / (2.0 * g)
y_max = y_burn + h_coast

print("Analytic burnout velocity (m/s):", v_burn_analytic)
print("Numeric burnout velocity (m/s):", v_burn_num)
print("Burnout altitude (m):", y_burn)
print("Coast height (m):", h_coast)
print("Maximum altitude (m):", y_max)
