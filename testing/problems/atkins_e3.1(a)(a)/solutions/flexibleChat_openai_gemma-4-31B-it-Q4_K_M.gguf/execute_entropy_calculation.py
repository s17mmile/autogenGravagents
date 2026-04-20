# filename: execute_entropy_calculation.py

def calculate_entropy_change(q_rev_joules, temperature_kelvin):
    """
    Calculate the change in entropy for a reversible isothermal process.

    Parameters:
    q_rev_joules (float): Heat transferred reversibly in joules.
    temperature_kelvin (float): Temperature in Kelvin.

    Returns:
    float: Change in entropy in J/K.
    """
    if temperature_kelvin <= 0:
        raise ValueError("Temperature must be greater than 0 K.")
    delta_s = q_rev_joules / temperature_kelvin
    return delta_s

# Given values
q_rev = 25000  # in joules
T = 273.15     # in kelvin

# Compute entropy change
entropy_change = calculate_entropy_change(q_rev, T)

# Output result
print(f"The change in entropy is {entropy_change:.2f} J/K")