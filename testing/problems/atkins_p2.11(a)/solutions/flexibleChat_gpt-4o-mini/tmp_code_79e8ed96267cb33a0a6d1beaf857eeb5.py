def calculate_temperature_rise(Q, m, c):
    # Calculate temperature rise
    Delta_T = Q / (m * c)
    return Delta_T

# Constants
Q = 10_000_000  # Heat produced in joules (10 MJ)
m = 65  # Mass of the human body in kg
c = 4184  # Specific heat capacity of water in J/(kg*K)

# Calculate and print the result
temperature_rise = calculate_temperature_rise(Q, m, c)
print(f'Temperature rise experienced by the body: {temperature_rise:.2f} K')