def calculate_kinetic_energy(voltage):
    """Calculate the kinetic energy of an electron given a voltage."""
    charge_of_electron = 1.602e-19  # Charge of electron in coulombs
    kinetic_energy = charge_of_electron * voltage
    return kinetic_energy

# Test the function with a voltage of 100 V
voltage = 100  # Voltage in volts
kinetic_energy = calculate_kinetic_energy(voltage)

# Print the result formatted to scientific notation
print(f'The kinetic energy of the electron is: {kinetic_energy:.2e} J')