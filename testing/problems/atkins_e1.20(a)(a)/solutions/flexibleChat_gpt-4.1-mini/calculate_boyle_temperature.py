# filename: calculate_boyle_temperature.py

def calculate_boyle_temperature(a, b, R):
    """
    Calculate the Boyle temperature for a gas using van der Waals constants.

    Parameters:
    a (float): van der Waals constant a (L^2 atm / mol^2), must be positive
    b (float): van der Waals constant b (L / mol), must be positive
    R (float): Universal gas constant (L atm / mol K), must be positive

    Returns:
    float: Boyle temperature in Kelvin
    """
    if a <= 0 or b <= 0 or R <= 0:
        raise ValueError("All input parameters must be positive numbers.")
    T_B = a / (b * R)
    return T_B


if __name__ == "__main__":
    # van der Waals constants for chlorine (Cl2)
    a = 6.49  # L^2 atm / mol^2
    b = 0.0562  # L / mol
    R = 0.08206  # L atm / (mol K)

    boyle_temperature = calculate_boyle_temperature(a, b, R)
    print(f"Approximate Boyle temperature of chlorine: {boyle_temperature:.2f} K")
