def calculate_heat_capacities(q, n, delta_T, R=8.314):
    # Check for valid inputs
    if n <= 0 or delta_T <= 0:
        raise ValueError('Amount of gas (n) and temperature change (delta_T) must be positive.')
    
    # Step 1: Calculate molar heat capacity at constant pressure (C_p)
    C_p = q / (n * delta_T)
    
    # Step 2: Calculate molar heat capacity at constant volume (C_v)
    C_v = C_p - R
    
    return C_p, C_v

# Given data
q = 229  # Heat supplied in Joules
n = 3.0  # Amount of gas in moles
Delta_T = 2.55  # Temperature change in Kelvin

# Calculate heat capacities
C_p, C_v = calculate_heat_capacities(q, n, Delta_T)

# Print the results
print(f'Molar heat capacity at constant pressure (C_p): {C_p:.2f} J/(mol K)')
print(f'Molar heat capacity at constant volume (C_v): {C_v:.2f} J/(mol K)')