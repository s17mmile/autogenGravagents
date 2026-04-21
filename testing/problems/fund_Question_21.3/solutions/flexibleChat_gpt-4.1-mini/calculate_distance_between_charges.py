# filename: calculate_distance_between_charges.py

def calculate_distance(q1_microC, q2_microC, force_N):
    """Calculate the distance between two point charges given their magnitudes in microcoulombs and the force in newtons."""
    # Coulomb's constant in N m^2 / C^2
    k = 8.988e9
    # Convert microcoulombs to coulombs
    q1 = q1_microC * 1e-6
    q2 = q2_microC * 1e-6
    # Calculate distance using Coulomb's law (magnitude of force depends on magnitude of charges)
    r = ((k * abs(q1 * q2)) / force_N) ** 0.5
    return r

# Given values
q1 = 26.0  # microcoulombs
q2 = -47.0 # microcoulombs (sign indicates attraction, magnitude used in calculation)
F = 5.70   # Newtons

distance = calculate_distance(q1, q2, F)
print(f"Distance between charges: {distance:.4f} meters")
