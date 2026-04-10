import math

# Constants
k = 1.38e-23  # Boltzmann constant in J/K
T = 20 + 273.15  # Temperature in Kelvin
sigma = 0.36e-18  # Collision cross-section in m^2
V = 100e-6  # Volume in m^3 (100 cm^3)

def calculate_pressure(k, T, sigma, V):
    try:
        # Calculate effective diameter from collision cross-section
        effective_diameter = math.sqrt(4 * sigma / math.pi)
        
        # Calculate diameter of the vessel (assuming spherical)
        vessel_diameter = (6 * V / math.pi) ** (1/3)
        
        # Calculate pressure when mean free path is comparable to vessel diameter
        pressure = (k * T) / (math.sqrt(2) * math.pi * effective_diameter**2 * vessel_diameter)
        return pressure
    except ZeroDivisionError:
        return "Error: Division by zero encountered in calculations."
    except Exception as e:
        return f"An error occurred: {e}"

# Calculate the pressure
calculated_pressure = calculate_pressure(k, T, sigma, V)

# Output the calculated pressure
print(f'The calculated pressure is: {calculated_pressure} Pa')