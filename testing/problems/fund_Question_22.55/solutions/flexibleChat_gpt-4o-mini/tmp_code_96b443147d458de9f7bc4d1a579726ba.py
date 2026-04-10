def calculate_electron_speed(distance_m, time_s):
    # Calculate acceleration using the formula a = 2s / t^2
    acceleration = (2 * distance_m) / (time_s ** 2)
    # Calculate final velocity using the formula v = at (initial velocity u = 0)
    final_velocity = acceleration * time_s
    return final_velocity

# Constants
distance_m = 0.02  # distance in meters (2.0 cm)
time_s = 1.5e-8  # time in seconds

# Calculate the speed of the electron
speed = calculate_electron_speed(distance_m, time_s)

# Print the final speed of the electron in both scientific and standard format
print(f'The speed of the electron as it strikes the second plate is approximately {speed:.2e} m/s')
print(f'The speed of the electron in standard format is approximately {speed:.2f} m/s')