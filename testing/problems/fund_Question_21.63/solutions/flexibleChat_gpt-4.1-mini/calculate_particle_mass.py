# filename: calculate_particle_mass.py

def calculate_mass():
    # Constants
    k = 8.99e9  # Coulomb's constant in N m^2 / C^2

    # Charges
    q1 = 30e-9  # Charge at origin in Coulombs
    q2 = -40e-9  # Charge at x=0.72 m in Coulombs
    q_particle = 42e-6  # Charge of particle in Coulombs

    # Positions in meters
    x1 = 0.0
    x2 = 0.72
    x_particle = 0.28

    # Distances
    r1 = x_particle - x1  # Distance from q1 to particle
    r2 = x2 - x_particle  # Distance from q2 to particle

    # Calculate force magnitudes
    F1_mag = k * abs(q1 * q_particle) / r1**2
    F2_mag = k * abs(q2 * q_particle) / r2**2

    # Determine force directions (signs)
    # Force from q1 on particle: both positive charges => repulsive => force on particle in +x direction
    F1 = F1_mag * 1  # positive x direction

    # Force from q2 on particle: q2 negative, particle positive => attractive => force towards q2 (positive x direction)
    F2 = F2_mag * 1  # positive x direction

    # Net force
    F_total = F1 + F2

    # Given acceleration
    a = 100000  # m/s^2

    # Calculate mass
    m = F_total / a

    return m

if __name__ == '__main__':
    mass = calculate_mass()
    print(f"The mass of the particle is {mass:.6e} kg")
