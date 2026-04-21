# filename: calculate_tire_pressure_function.py

def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def calculate_pressure(P1, T1_Celsius, T2_Celsius):
    """
    Calculate the final pressure of a gas at constant volume using Gay-Lussac's Law.
    Parameters:
        P1 (float): Initial pressure in lb/in^2 (must be > 0)
        T1_Celsius (float): Initial temperature in Celsius (must be > -273.15)
        T2_Celsius (float): Final temperature in Celsius (must be > -273.15)
    Returns:
        float: Final pressure in lb/in^2
    Raises:
        ValueError: If inputs are out of physical bounds.
    """
    if P1 <= 0:
        raise ValueError("Initial pressure must be greater than zero.")
    if T1_Celsius <= -273.15 or T2_Celsius <= -273.15:
        raise ValueError("Temperatures must be above absolute zero (-273.15 Celsius).")

    T1 = celsius_to_kelvin(T1_Celsius)
    T2 = celsius_to_kelvin(T2_Celsius)

    P2 = P1 * (T2 / T1)
    return P2


# Example usage
if __name__ == "__main__":
    initial_pressure = 24.0  # lb/in^2
    initial_temp = -5.0      # Celsius
    final_temp = 35.0        # Celsius

    final_pressure = calculate_pressure(initial_pressure, initial_temp, final_temp)
    print(f"The pressure on the summer day is {final_pressure:.2f} lb/in^2")
