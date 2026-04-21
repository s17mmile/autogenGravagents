# filename: calculate_bandgap_tio2.py

def calculate_bandgap(wavelength_nm=350):
    """
    Calculate the band gap energy in electronvolts for a given wavelength in nanometers.

    Parameters:
    wavelength_nm (float): Wavelength in nanometers. Must be positive.

    Returns:
    float: Energy in electronvolts corresponding to the wavelength.
    """
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be a positive number.")

    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    c = 3.0e8      # Speed of light in m/s
    e_charge = 1.602e-19  # Elementary charge in Coulombs (J/eV conversion)

    # Convert wavelength from nm to meters
    wavelength_m = wavelength_nm * 1e-9

    # Calculate energy in Joules
    energy_joules = (h * c) / wavelength_m

    # Convert energy to electronvolts
    energy_ev = energy_joules / e_charge

    return energy_ev

# Calculate band gap energy for 350 nm
band_gap_ev = calculate_bandgap(350)

# Print the result to console
print(f'Band gap energy for TiO2 at 350 nm wavelength: {band_gap_ev:.3f} eV')

# Save result to a file
with open('band_gap_result.txt', 'w') as f:
    f.write(f'Band gap energy for TiO2 at 350 nm wavelength: {band_gap_ev:.3f} eV\n')
