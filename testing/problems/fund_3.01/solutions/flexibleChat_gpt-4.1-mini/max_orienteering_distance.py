# filename: max_orienteering_distance.py
import math

# Given vectors
# a: 2.0 km due east
ax, ay = 2.0, 0.0

# b: 2.0 km at 30 degrees north or south of east
b_magnitude = 2.0
b_angle_deg = 30
b_angle_rad = math.radians(b_angle_deg)

# c: 1.0 km due west or due east
c_magnitude = 1.0

# Possible sign options for b_y and c_x
b_y_options = [b_magnitude * math.sin(b_angle_rad), -b_magnitude * math.sin(b_angle_rad)]
c_x_options = [c_magnitude, -c_magnitude]

# b_x is always positive (cos 30 degrees)
bx = b_magnitude * math.cos(b_angle_rad)

max_distance = 0.0

for by in b_y_options:
    for cx in c_x_options:
        Rx = ax + bx + cx
        Ry = ay + by
        distance = math.sqrt(Rx**2 + Ry**2)
        if distance > max_distance:
            max_distance = distance

print(f"Greatest distance from base camp after three moves: {max_distance:.4f} km")

# The output is the maximum straight-line distance achievable after the three displacements.