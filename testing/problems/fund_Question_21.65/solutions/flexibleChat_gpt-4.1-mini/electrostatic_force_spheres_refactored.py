# filename: electrostatic_force_spheres_refactored.py

# Constants
k = 8.99e9  # Coulomb's constant in N m^2 / C^2
Q = 2.00e-14  # Initial charge unit in Coulombs

d = 1.20  # Distance between spheres A and B in meters


def redistribute_charge(q1, q2):
    """Redistribute charge equally between two identical metal spheres after contact."""
    total_charge = q1 + q2
    shared_charge = total_charge / 2
    return shared_charge, shared_charge


def calculate_coulomb_force(q1, q2, distance):
    """Calculate magnitude of electrostatic force between two point charges."""
    return k * abs(q1 * q2) / distance**2


# Initial charges
Q_A = Q
Q_B = -Q / 4
Q_C = Q / 2

# Sphere C touched to Sphere A
Q_A, Q_C = redistribute_charge(Q_A, Q_C)

# Sphere C touched to Sphere B
Q_B, Q_C = redistribute_charge(Q_B, Q_C)

# Calculate force magnitude
F = calculate_coulomb_force(Q_A, Q_B, d)

# Output results
print(f"Final charge on Sphere A: {Q_A:.3e} C")
print(f"Final charge on Sphere B: {Q_B:.3e} C")
print(f"Magnitude of electrostatic force between spheres A and B: {F:.3e} N")

# Return all final charges and force
Q_A, Q_B, Q_C, F
