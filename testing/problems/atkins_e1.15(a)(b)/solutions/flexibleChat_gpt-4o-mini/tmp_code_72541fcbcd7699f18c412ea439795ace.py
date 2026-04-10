def calculate_molar_volume(R, T, P):
    # Check for valid pressure
    if P <= 0:
        raise ValueError('Pressure must be greater than zero.')
    
    # Step 1: Calculate the ideal molar volume using the ideal gas law
    V_ideal = (R * T) / P
    
    # Step 2: Calculate the actual molar volume (12% smaller than ideal)
    V_actual = V_ideal * (1 - 0.12)
    
    return V_actual

# Constants
R = 0.0821  # Ideal gas constant in L·atm/(K·mol)
T = 250    # Temperature in Kelvin
P = 15     # Pressure in atm

# Calculate the molar volume
molar_volume = calculate_molar_volume(R, T, P)

# Output the actual molar volume with two significant figures
print(f'The molar volume of the gas is approximately {molar_volume:.2f} L')