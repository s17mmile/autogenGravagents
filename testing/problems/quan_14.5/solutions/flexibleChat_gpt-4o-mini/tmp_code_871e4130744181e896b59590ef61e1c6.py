def calculate_dipole_moment():
    # Constants
    ionization_energy_Na = 5.14  # eV
    electron_affinity_Cl = 3.61    # eV
    R_e = 2.36e-10                  # m (equilibrium distance)
    q = 1.602e-19                   # C (elementary charge)
    conversion_factor = 3.33564e-30 # C m / D (conversion to Debye)

    # Calculate energy change for ion formation
    Delta_E = ionization_energy_Na - electron_affinity_Cl

    # Calculate dipole moment in C m
    D = q * R_e

    # Convert dipole moment to Debye
    D_e = D / conversion_factor

    # Output the result
    print(f'Dipole moment D_e for NaCl: {D_e:.2f} D')

# Call the function to perform the calculation
calculate_dipole_moment()