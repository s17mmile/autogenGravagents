import math

# Constants
P0 = 101325  # Sea level pressure in Pascals
M_N2 = 0.028  # Molar mass of nitrogen in kg/mol
M_O2 = 0.032  # Molar mass of oxygen in kg/mol
R = 8.314  # Universal gas constant in J/(mol*K)
T = 298  # Temperature in Kelvin (25 degrees Celsius)

def calculate_height_and_pressure():
    # Calculate the height where the atmosphere is 90% nitrogen and 10% oxygen
    ratio = 9  # P_N2 / P_O2 = 9
    try:
        # Solve for height h
        h = (R * T / 9.81) * math.log(ratio * (M_O2 / M_N2))
        # Calculate the effective molar mass of the air at that height
        M_eff = 0.90 * M_N2 + 0.10 * M_O2
        # Calculate the pressure at that height using the barometric formula
        P_h = P0 * math.exp(-((M_eff * 9.81 * h) / (R * T)))
        return h, P_h
    except ValueError as e:
        return str(e)

# Main execution
height, pressure = calculate_height_and_pressure()
print(f'Height where atmosphere is 90% nitrogen and 10% oxygen: {height:.2f} meters')
print(f'Pressure at that height: {pressure:.2f} Pascals')