def calculate_diving_bell_volume(P0=101325, rho=1025, g=9.81, h=50, V1=3.0):
    # Step 1: Calculate pressure at depth
    P2 = P0 + (rho * g * h)  # Total pressure at depth in Pascals
    
    # Error handling for division by zero
    if P2 == 0:
        raise ValueError('Pressure at depth cannot be zero.')
    
    # Step 2: Calculate new volume using Boyle's Law
    V2 = (P0 * V1) / P2  # Volume at depth
    
    return V2

# Call the function and output the result
volume = calculate_diving_bell_volume()
print(f'The volume of the air space in the diving bell at a depth of 50 meters is approximately {volume:.3f} m^3.')