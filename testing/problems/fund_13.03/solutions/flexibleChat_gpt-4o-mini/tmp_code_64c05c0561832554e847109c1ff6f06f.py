import math

# Constants
G = 6.674 * 10**-11  # gravitational constant in m^3/(kg*s^2)
M = 5.972 * 10**24   # mass of Earth in kg
R_E = 6.371 * 10**6   # radius of Earth in meters
m = 1  # mass of the asteroid (arbitrary, cancels out)

# Initial conditions
r_i = 10 * R_E       # initial distance from Earth's center in meters
v_i = 12000          # initial speed of the asteroid in m/s

# Initial energies
KE_i = 0.5 * (v_i ** 2)  # initial kinetic energy
PE_i = - (G * M * m) / r_i  # initial potential energy

# Final conditions
r_f = R_E  # final distance from Earth's center in meters

# Final potential energy
PE_f = - (G * M * m) / r_f  # final potential energy

# Conservation of energy
# KE_i + PE_i = KE_f + PE_f
# 0.5 * v_f^2 = KE_i + PE_i - PE_f
final_speed_squared = 2 * (KE_i + PE_i - PE_f)

# Check for non-negative value before taking square root
if final_speed_squared < 0:
    raise ValueError('Calculated speed squared is negative, check energy calculations.')

# Final speed
v_f = math.sqrt(final_speed_squared)

# Convert to km/s
v_f_km_s = v_f / 1000

# Output the final speed
print(f'The final speed of the asteroid is {v_f_km_s:.2f} km/s')