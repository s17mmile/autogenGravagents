# filename: calculate_electrons_removed.py

def electrons_removed(charge_coulombs):
    """Calculate the number of electrons removed to leave a given positive charge.

    Args:
        charge_coulombs (float): The positive charge in coulombs.

    Returns:
        int: Number of electrons removed (rounded to nearest integer).
    """
    # Elementary charge of one electron in coulombs
    elementary_charge = 1.602e-19
    
    if charge_coulombs <= 0:
        raise ValueError("Charge must be positive to represent electrons removed.")
    
    number_of_electrons = charge_coulombs / elementary_charge
    return int(round(number_of_electrons))

# Given charge
charge = 1.0e-7  # in coulombs

# Calculate number of electrons removed
num_electrons = electrons_removed(charge)
print(f"Number of electrons removed to leave a charge of {charge} C: {num_electrons}")
