def test_force_with_different_distances(distances):
    # Constants
    k = 8.99e9  # Coulomb's constant in N m^2/C^2
    alpha_particle_charge = 2 * 1.602e-19  # Charge of alpha particle in Coulombs
    gold_nucleus_charge = 79 * 1.602e-19  # Charge of gold nucleus in Coulombs

    for distance in distances:
        # Check for division by zero
        if distance == 0:
            raise ValueError('Distance cannot be zero.')  

        # Calculate the force using Coulomb's law
        force = k * (alpha_particle_charge * gold_nucleus_charge) / (distance ** 2)

        # Print the result for each distance
        print(f'Distance: {distance:.2e} m, Force: {force:.2e} N')

# Test the function with a range of distances
test_distances = [0.00300e-10, 0.01e-10, 0.1e-10, 1e-10]  # Distances in meters

test_force_with_different_distances(test_distances)