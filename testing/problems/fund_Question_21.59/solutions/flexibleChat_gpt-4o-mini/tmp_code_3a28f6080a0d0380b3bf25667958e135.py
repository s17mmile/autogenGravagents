def calculate_total_charge(mass_of_electrons_kg):
    # Constants
    mass_of_electron = 9.11e-31  # kg
    charge_of_electron = -1.602e-19  # coulombs

    # Error handling
    if mass_of_electrons_kg <= 0:
        raise ValueError('Mass of electrons must be a positive number.')

    # Step 1: Calculate the number of electrons
    number_of_electrons = mass_of_electrons_kg / mass_of_electron

    # Step 2: Calculate the total charge
    total_charge = number_of_electrons * charge_of_electron

    return total_charge

# Calculate and print the total charge for 75.0 kg of electrons
mass_of_electrons = 75.0  # kg
charge = calculate_total_charge(mass_of_electrons)
print(f'The total charge of {mass_of_electrons} kg of electrons is approximately {charge:.2e} C')