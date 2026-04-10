import math

# Given data
m = 1000  # mass of the automobile in kg
A_road = 0.05  # amplitude of the road bumps in meters
v = 20 * 1000 / 3600  # speed in m/s
lambda_ = 0.2  # wavelength in meters

# Calculate angular frequency
omega = (v / lambda_) * (2 * math.pi)  # angular frequency in rad/s

# Calculate the peak driving force
F_0 = m * A_road * omega**2

# Output the result
F_0