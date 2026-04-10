import math

# Constants
p0 = 1.0  # sea level pressure in atm
M = 0.029  # molar mass of air in kg/mol
g = 9.81  # acceleration due to gravity in m/s^2
R = 8.314  # universal gas constant in J/(mol K)

# Function to estimate temperature based on altitude
def estimate_temperature(altitude):
    # Standard atmosphere model: temperature decreases with altitude
    T0 = 288.15  # sea level standard temperature in K
    lapse_rate = 0.0065  # temperature lapse rate in K/m
    return T0 - lapse_rate * altitude

# Altitude in meters
h = 11000  # cruising altitude

# Calculate average temperature at cruising altitude
T = estimate_temperature(h)

# Calculate the exponent
exponent = -(M * g / (R * T)) * h

# Calculate pressure at altitude
p = p0 * math.exp(exponent)

# Convert pressure to hPa (1 atm = 1013.25 hPa)
pressure_hPa = p * 1013.25

# Print results
print(f'Pressure at {h/1000} km: {pressure_hPa:.2f} hPa')
print(f'Pressure at {h/1000} km: {p:.6f} atm')