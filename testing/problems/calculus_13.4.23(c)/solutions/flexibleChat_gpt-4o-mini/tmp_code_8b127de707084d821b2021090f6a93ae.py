import math

# Constants
initial_speed = 200  # m/s
angle = 60  # degrees
gravity = 9.81  # m/s^2

# Convert angle to radians
angle_rad = math.radians(angle)

# Calculate horizontal and vertical components of initial velocity
Vx = initial_speed * math.cos(angle_rad)
Vy = initial_speed * math.sin(angle_rad)

# Calculate time of flight (total time until it returns to the same vertical level)
T = (2 * Vy) / gravity

# Calculate final vertical velocity just before impact
Vy_final = Vy + gravity * T  # Corrected formula

# Calculate speed at impact using Pythagorean theorem
speed_at_impact = math.sqrt(Vx**2 + Vy_final**2)

# Output the result
print(f'Speed at impact: {speed_at_impact:.2f} m/s')