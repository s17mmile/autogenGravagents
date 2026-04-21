# filename: net_electrostatic_force.py

def calculate_net_force():
    # Constants
    k = 8.988e9  # Coulomb's constant in N m^2 / C^2

    # Given data
    Q_bee = 45.0e-12  # Charge on bee in Coulombs
    q_near = -1.00e-12  # Induced charge on near side of pollen grain in Coulombs
    q_far = 1.00e-12    # Induced charge on far side of pollen grain in Coulombs

    R_bee = 0.005  # Radius of bee in meters (1.000 cm diameter)
    d_pollen = 40.0e-6  # Diameter of pollen grain in meters

    # Distances from bee center to near and far induced charges
    r_near = R_bee
    r_far = R_bee + d_pollen

    # Calculate forces on each induced charge (including sign for direction)
    F_near = q_near * k * Q_bee / (r_near ** 2)
    F_far = q_far * k * Q_bee / (r_far ** 2)

    # Net force is sum of forces (considering direction)
    F_net = F_near + F_far

    # Return magnitude of net force
    return abs(F_net)

if __name__ == '__main__':
    net_force = calculate_net_force()
    print(f"Magnitude of net electrostatic force on pollen grain: {net_force:.3e} N")
