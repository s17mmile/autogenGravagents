def calculate_mole_fraction(moles_nitrogen, moles_argon):
    # Constants
    molar_mass_nitrogen = 28.02  # g/mol
    molar_mass_argon = 39.95  # g/mol

    # Calculate total mass of the mixture
    mass_nitrogen = moles_nitrogen * molar_mass_nitrogen
    mass_argon = moles_argon * molar_mass_argon
    total_mass = mass_nitrogen + mass_argon

    # Calculate mole fraction of argon
    total_moles = moles_nitrogen + moles_argon
    if total_moles == 0:
        raise ValueError('Total moles cannot be zero for mole fraction calculation.')
    mole_fraction_argon = moles_argon / total_moles

    # Output results
    print(f'Total mass of the atmospheric nitrogen sample: {total_mass:.2f} g')
    print(f'Mole fraction of argon in the atmospheric nitrogen sample: {mole_fraction_argon:.4f}')

# Known composition of atmospheric air
moles_nitrogen = 99.07  # moles of N2 in 100 moles of air
moles_argon = 0.93  # moles of Ar in 100 moles of air

# Call the function to perform calculations
calculate_mole_fraction(moles_nitrogen, moles_argon)