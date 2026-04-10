import math

# Constants
M_sun = 1.989e30  # Mass of the Sun in kg
G = 6.674e-11     # Gravitational constant in m^3 kg^-1 s^-2
light_year = 9.461e15  # 1 light year in meters
seconds_per_year = 3.154e7  # Number of seconds in a year

def calculate_period_of_revolution(separation_light_years):
    # Convert separation from light years to meters
    separation_meters = separation_light_years * light_year  
    # Total mass of the two stars
    M_total = 2 * M_sun
    # Applying Kepler's Third Law
    T_squared = (4 * math.pi**2) / (G * M_total) * (separation_meters**3)
    T = math.sqrt(T_squared)  # Period in seconds
    # Convert period from seconds to years
    T_years = T / seconds_per_year  # Convert to years
    return T_years

# Given value
separation_light_years = 4  # Separation in light years
# Calculate the period
period = calculate_period_of_revolution(separation_light_years)
# Output the period
print(f'The calculated period of revolution for the double stars is approximately {period:.2f} years.')