def calculate_corrected_force_on_alpha_particle():
    # Constants
    k = 8.99e9  # Coulomb's constant in N m^2/C^2
    alpha_particle_charge = 2 * 1.602e-19  # Charge of alpha particle in Coulombs
    gold_nucleus_charge = 79 * 1.602e-19  # Charge of gold nucleus in Coulombs
    distance = 0.00300e-10  # Distance in meters (0.00300 Å)

    # Check for division by zero
    if distance == 0:
        raise ValueError('Distance cannot be zero.')  

    # Calculate the force using Coulomb's law
    force = k * (alpha_particle_charge * gold_nucleus_charge) / (distance ** 2)

    # Print the result
    print(f'The calculated electrostatic force on the alpha particle due to the gold nucleus is approximately {force:.2e} N')

# Call the function to perform the corrected calculation
calculate_corrected_force_on_alpha_particle()