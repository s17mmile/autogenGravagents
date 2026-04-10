import math

# Constants
c_w = 0.75  # Drag coefficient
rho = 1.225  # Air density in kg/m^3
burn_time = 1.5  # Burn time in seconds
initial_mass = 0.054  # Initial mass of the rocket in kg
propellant_mass = 0.011  # Mass of the propellant in kg
final_mass = initial_mass - propellant_mass  # Final mass of the rocket

# Rocket parameters
exhaust_speed = 800  # Exhaust speed in m/s
impulse = 8.5  # Total impulse in N-s

def calculate_cross_sectional_area(diameter):
    return math.pi * (diameter / 2) ** 2


def calculate_drag_force(c_w, rho, A, v):
    return 0.5 * c_w * rho * A * (v ** 2)


def calculate_work_done(F_d, distance):
    return F_d * distance

# Step 1: Calculate the cross-sectional area A
diameter = 0.024  # Diameter in meters
A = calculate_cross_sectional_area(diameter)

# Step 2: Calculate the average velocity during the burn time
v_avg = exhaust_speed / 2  # Assuming linear acceleration

# Step 3: Calculate the drag force F_d
F_d = calculate_drag_force(c_w, rho, A, v_avg)

# Step 4: Calculate the distance traveled during the burn time
distance = v_avg * burn_time

# Step 5: Calculate the work done against drag W_d
W_d = calculate_work_done(F_d, distance)

# Step 6: Adjust the speed calculation
adjusted_speed = (impulse - W_d) / (final_mass / burn_time)

# Output results
print(f'Drag Force: {F_d:.2f} N')
print(f'Work Done Against Drag: {W_d:.2f} J')
print(f'Adjusted Speed at Burnout: {adjusted_speed:.2f} m/s')