# filename: hydrogen_ground_state_energy.py

def calculate_hydrogen_ground_state_energy():
    """Calculate the ground-state energy of the hydrogen atom in joules and electronvolts."""
    # Physical constants in SI units
    m_e = 9.10938356e-31  # electron mass in kg
    e = 1.602176634e-19   # elementary charge in C
    epsilon_0 = 8.854187817e-12  # vacuum permittivity in F/m
    h = 6.62607015e-34    # Planck's constant in J*s
    n = 1  # principal quantum number for ground state

    # Calculate ground state energy using Bohr model formula
    E_n = -(m_e * e**4) / (8 * epsilon_0**2 * h**2 * n**2)  # energy in joules

    # Convert energy from joules to electronvolts
    joule_to_eV = 1 / e
    E_n_eV = E_n * joule_to_eV

    return E_n, E_n_eV

if __name__ == '__main__':
    energy_joules, energy_eV = calculate_hydrogen_ground_state_energy()
    print(f"Ground-state energy of hydrogen atom:")
    print(f"{energy_joules:.4e} Joules")
    print(f"{energy_eV:.4f} electronvolts")
