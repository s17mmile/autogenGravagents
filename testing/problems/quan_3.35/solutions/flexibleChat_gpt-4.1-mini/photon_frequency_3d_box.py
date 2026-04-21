# filename: photon_frequency_3d_box.py

def calculate_photon_frequency():
    """Calculate the frequency of the photon emitted during the transition from the lowest excited state to the ground state in a 3D box."""
    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    m = 9.109e-31  # Electron mass in kg
    angstrom_to_m = 1e-10  # Conversion factor from Angstrom to meters

    # Box dimensions in meters
    Lx = 5.00 * angstrom_to_m
    Ly = 3.00 * angstrom_to_m
    Lz = 6.00 * angstrom_to_m

    # Quantum numbers for ground state
    n_ground = (1, 1, 1)

    # Function to calculate energy for given quantum numbers
    def energy(n_x, n_y, n_z):
        return (h**2 / (8 * m)) * ((n_x**2) / (Lx**2) + (n_y**2) / (Ly**2) + (n_z**2) / (Lz**2))

    # Calculate ground state energy
    E_ground = energy(*n_ground)

    # Possible lowest excited states by increasing one quantum number by 1
    excited_states = [
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2)
    ]

    # Calculate energies of excited states
    energies_excited = [(state, energy(*state)) for state in excited_states]

    # Find the lowest excited state energy
    state_lowest_excited, E_lowest_excited = min(energies_excited, key=lambda x: x[1])

    # Energy difference
    delta_E = E_lowest_excited - E_ground

    # Frequency of emitted photon
    frequency = delta_E / h

    return frequency

# Calculate and print the frequency
freq = calculate_photon_frequency()
print(f"Frequency of emitted photon: {freq:.3e} Hz")
