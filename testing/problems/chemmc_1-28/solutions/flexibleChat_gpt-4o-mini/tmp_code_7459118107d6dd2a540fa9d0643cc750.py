def calculate_lyman_series_limit():
    """
    Calculate the energy and wavelength of a photon associated with the series limit of the Lyman series in hydrogen.
    """
    # Constants
    R_H = 13.6  # Rydberg constant in eV
    h = 6.626e-34  # Planck's constant in J*s
    c = 3.00e8  # Speed of light in m/s
    conversion_factor = 1.602e-19  # eV to J conversion

    try:
        # Energy calculation for the series limit of the Lyman series
        n1 = 1  # Ground state
        n2 = float('inf')  # Series limit
        energy_eV = R_H * (1/n1**2 - 1/n2**2)  # Energy in eV

        # Convert energy to Joules
        energy_J = energy_eV * conversion_factor

        # Wavelength calculation
        wavelength_m = (h * c) / energy_J  # Wavelength in meters
        wavelength_nm = wavelength_m * 1e9  # Convert to nanometers

        # Print results with formatted output
        print(f'Energy of the photon: {energy_eV:.2f} eV')
        print(f'Wavelength of the photon: {wavelength_nm:.2f} nm')
    except Exception as e:
        print(f'An error occurred: {e}')  

# Call the function to execute the calculations
calculate_lyman_series_limit()