def calculate_molar_volume(P, T):
    # Constants for CO2
    R = 0.0821  # Ideal gas constant in L·atm/(K·mol)
    a = 3.59  # L^2 atm / mol^2
    b = 0.0427  # L / mol

    # Error handling for invalid inputs
    if P <= 0 or T <= 0:
        raise ValueError('Pressure and temperature must be positive values.')

    # Calculate the terms
    RT_over_P = (R * T) / P  # RT/P
    b_term = b  # b
    a_over_P = a / P  # a/P

    # Calculate molar volume using the corrected equation
    V_m = RT_over_P - b_term + a_over_P
    return V_m

# Given values
P = 100  # Pressure in atm
T = 500  # Temperature in K

# Calculate and output the molar volume
molar_volume = calculate_molar_volume(P, T)
print(f'The estimated molar volume of CO2 at {T} K and {P} atm is approximately {molar_volume:.4f} L/mol.')