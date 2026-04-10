import scipy.constants as const

# Calculate the ionization energy of krypton based on ultraviolet radiation

def calculate_ionization_energy():
    """
    This function calculates the ionization energy of krypton when exposed to ultraviolet radiation.
    It uses the wavelength of the radiation and the speed of the ejected electrons to compute the energy.
    """
    # Constants
    wavelength_nanometers = 58.4  # wavelength in nanometers
    wavelength_m = wavelength_nanometers * 1e-9  # convert to meters
    speed_of_electron = 1.59e6  # speed of ejected electron in m/s
    mass_of_electron = const.electron_mass  # mass of electron in kg

    # Step 1: Calculate the energy of the incident photon (E)
    E = (const.h * const.c) / wavelength_m

    # Step 2: Calculate the kinetic energy of the ejected electron (K.E.)
    KE = 0.5 * mass_of_electron * speed_of_electron**2

    # Step 3: Calculate the ionization energy (phi)
    ionization_energy_joules = E - KE

    # Step 4: Convert ionization energy to electronvolts (eV)
    ionization_energy_eV = ionization_energy_joules / const.e

    # Output the ionization energy
    print(f'Ionization Energy of Krypton: {ionization_energy_eV:.2f} eV')

# Call the function to execute the calculation
calculate_ionization_energy()