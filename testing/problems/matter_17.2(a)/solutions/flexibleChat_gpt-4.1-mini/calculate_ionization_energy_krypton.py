# filename: calculate_ionization_energy_krypton.py

def calculate_ionization_energy(wavelength_nm, electron_speed_m_s):
    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    c = 3.00e8     # Speed of light in m/s
    m_e = 9.11e-31 # Electron mass in kg
    e_charge = 1.602e-19  # Elementary charge in C (for eV conversion)

    # Input validation
    if wavelength_nm <= 0:
        raise ValueError('Wavelength must be positive')
    if electron_speed_m_s < 0:
        raise ValueError('Electron speed cannot be negative')

    # Convert wavelength from nm to meters
    wavelength_m = wavelength_nm * 1e-9

    # Calculate photon energy (Joules)
    E_photon = h * c / wavelength_m

    # Calculate kinetic energy of ejected electron (Joules)
    KE = 0.5 * m_e * electron_speed_m_s**2

    # Calculate ionization energy (Joules)
    IE_joules = E_photon - KE
    if IE_joules < 0:
        raise ValueError('Calculated ionization energy is negative, check inputs')

    # Convert ionization energy to electronvolts
    IE_eV = IE_joules / e_charge

    return IE_eV

# Given values
wavelength = 58.4  # nm
electron_speed = 1.59e6  # m/s

# Calculate ionization energy
ionization_energy = calculate_ionization_energy(wavelength, electron_speed)

# Save result to a file
with open('ionization_energy_krypton.txt', 'w') as f:
    f.write(f'Ionization energy of krypton: {ionization_energy:.2f} eV\n')
