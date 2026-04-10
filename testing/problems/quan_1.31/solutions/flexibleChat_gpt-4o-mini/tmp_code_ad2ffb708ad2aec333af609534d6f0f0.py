def debug_force_calculation():
    # Constants
    k = 8.99e9  # Coulomb's constant in N m^2/C^2
    alpha_particle_charge = 2 * 1.602e-19  # Charge of alpha particle in Coulombs
    gold_nucleus_charge = 79 * 1.602e-19  # Charge of gold nucleus in Coulombs
    distance = 0.00300e-10  # Distance in meters (0.00300 Å)

    # Check for division by zero
    if distance == 0:
        raise ValueError('Distance cannot be zero.')  

    # Calculate the numerator and denominator
    numerator = alpha_particle_charge * gold_nucleus_charge
    denominator = distance ** 2

    # Print intermediate values for debugging
    print(f'Alpha Particle Charge: {alpha_particle_charge:.2e} C')
    print(f'Gold Nucleus Charge: {gold_nucleus_charge:.2e} C')
    print(f'Distance: {distance:.2e} m')
    print(f'Numerator: {numerator:.2e} C^2')
    print(f'Denominator: {denominator:.2e} m^2')

    # Calculate the force using Coulomb's law
    force = k * numerator / denominator

    # Print the result
    print(f'The calculated electrostatic force on the alpha particle due to the gold nucleus is approximately {force:.2e} N')

# Call the function to perform the debug calculation
debug_force_calculation()