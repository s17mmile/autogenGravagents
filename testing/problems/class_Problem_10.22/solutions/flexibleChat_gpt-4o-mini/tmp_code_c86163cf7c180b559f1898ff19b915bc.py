import math

# Constants
height = 27  # height in meters
g = 9.81  # acceleration due to gravity in m/s^2
omega = 7.2921e-5  # angular velocity of the Earth in rad/s

# Step 1: Calculate the velocity of the falling lead
velocity = math.sqrt(2 * g * height)  # velocity just before impact

# Step 2: Calculate the Coriolis acceleration
coriolis_acceleration = 2 * omega * velocity  # Coriolis acceleration

# Step 3: Calculate the time of fall
time_of_fall = math.sqrt(2 * height / g)  # time of fall

# Step 4: Calculate the horizontal deflection due to Coriolis effect
deflection_distance = 0.5 * coriolis_acceleration * time_of_fall**2  # deflection distance

# Output the result
# Note: The Coriolis effect is minimal for a drop height of 27 meters.
print(f'The lead bullets landed approximately {deflection_distance * 1000:.2f} mm from the direct vertical.')