# filename: electrostatic_force_particle3_improved.py
import numpy as np

def calculate_electrostatic_force():
    """
    Calculate the net electrostatic force on particle 3 due to particles 1 and 2.
    Returns:
        np.ndarray: Net force vector on particle 3 in Newtons.
    """
    # Coulomb's constant in N m^2 / C^2
    k_e = 8.988e9

    # Charges in Coulombs
    Q1 = 80.0e-9
    Q2 = 80.0e-9
    q = 18.0e-9

    # Positions in meters
    r1 = np.array([0.0, 3.00e-3])  # Particle 1 position
    r2 = np.array([0.0, -3.00e-3]) # Particle 2 position
    r3 = np.array([4.00e-3, 0.0])  # Particle 3 position

    # Vector from particle 1 to particle 3
    r_31 = r3 - r1
    r_31_mag = np.linalg.norm(r_31)
    if r_31_mag == 0:
        raise ValueError("Particles 1 and 3 are at the same position, force is undefined.")
    r_31_hat = r_31 / r_31_mag

    # Vector from particle 2 to particle 3
    r_32 = r3 - r2
    r_32_mag = np.linalg.norm(r_32)
    if r_32_mag == 0:
        raise ValueError("Particles 2 and 3 are at the same position, force is undefined.")
    r_32_hat = r_32 / r_32_mag

    # Calculate force magnitudes using Coulomb's law
    F_31_mag = k_e * Q1 * q / r_31_mag**2
    F_32_mag = k_e * Q2 * q / r_32_mag**2

    # Calculate force vectors
    F_31 = F_31_mag * r_31_hat
    F_32 = F_32_mag * r_32_hat

    # Net force on particle 3
    F_net = F_31 + F_32

    # Print results with units
    print(f"Electrostatic force on particle 3:")
    print(f"  x-component: {F_net[0]:.3e} N")
    print(f"  y-component: {F_net[1]:.3e} N")

    # Calculate magnitude and direction (angle in degrees)
    F_net_mag = np.linalg.norm(F_net)
    F_net_angle = np.degrees(np.arctan2(F_net[1], F_net[0]))
    print(f"  Magnitude: {F_net_mag:.3e} N")
    print(f"  Direction: {F_net_angle:.2f} degrees from +x axis")

    return F_net

# Execute the function
calculate_electrostatic_force()
