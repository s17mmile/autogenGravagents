# filename: copper_coin_repulsion_improved.py

def calculate_repulsive_force(atoms_per_coin=3e22, protons_per_atom=29, difference_percent=0.00010, distance=1.0):
    """
    Calculate the repulsive force between two copper coins due to a slight difference in proton and electron charge magnitudes.

    Parameters:
    atoms_per_coin (float): Number of copper atoms per coin (unitless)
    protons_per_atom (int): Number of protons (and electrons) per copper atom (unitless)
    difference_percent (float): Percentage difference between proton and electron charge magnitudes (percent)
    distance (float): Distance between the two coins in meters (m)

    Returns:
    tuple: (force in Newtons (N), net charge per atom in Coulombs (C), net charge per coin in Coulombs (C))
    """
    # Constants
    elementary_charge = 1.602176634e-19  # Elementary charge in Coulombs (C)
    k = 8.9875517923e9  # Coulomb's constant in N m^2 / C^2

    # Convert difference percentage to fraction
    difference_fraction = difference_percent / 100  # unitless

    # Electron charge magnitude (C)
    electron_charge = elementary_charge * (1 - difference_fraction)  # C

    # Net charge per atom (C)
    # Each atom has protons_per_atom protons (+ charge) and electrons (- charge)
    net_charge_per_atom = protons_per_atom * (elementary_charge - electron_charge)  # C

    # Net charge per coin (C)
    net_charge_per_coin = atoms_per_coin * net_charge_per_atom  # C

    # Calculate force using Coulomb's law (N)
    force = k * (net_charge_per_coin ** 2) / (distance ** 2)  # N

    return force, net_charge_per_atom, net_charge_per_coin

if __name__ == "__main__":
    force, q_atom, q_coin = calculate_repulsive_force()
    print(f"Net charge per atom: {q_atom:.3e} C")
    print(f"Net charge per coin: {q_coin:.3e} C")
    print(f"Repulsive force between two copper coins: {force:.3e} N")
