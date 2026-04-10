import math

# Constants
M_sun = 1.989e30  # Mass of the Sun in kg
G = 6.674e-11     # Gravitational constant in m^3 kg^-1 s^-2
light_year = 9.461e15  # 1 light year in meters

# Given values
separation_light_years = 4  # Separation in light years
separation_meters = separation_light_years * light_year  # Convert to meters

# Total mass of the two stars
M_total = 2 * M_sun

# Step-by-step calculations
# 1. Calculate the denominator
denominator = G * M_total

# 2. Calculate the numerator
numerator = 4 * math.pi**2

# 3. Calculate (separation_meters)^3
separation_cubed = separation_meters**3

# 4. Apply Kepler's Third Law
T_squared = (numerator / denominator) * separation_cubed
T = math.sqrt(T_squared)  # Period in seconds

# Convert period from seconds to years
seconds_per_year = 3.154e7  # Number of seconds in a year
T_years = T / seconds_per_year  # Convert to years

# Output the period
print(f'The calculated period of revolution for the double stars is approximately {T_years:.2f} years.')