# filename: calculate_final_temperature.py

def calculate_final_temperature(mass_g, initial_temp_K, current_A, resistance_ohm, time_s):
    """Calculate the final temperature of water heated by an immersion heater.

    Parameters:
    mass_g (float): Mass of water in grams
    initial_temp_K (float): Initial temperature in Kelvin
    current_A (float): Current in Amperes
    resistance_ohm (float): Resistance in Ohms
    time_s (float): Time in seconds

    Returns:
    tuple: Final temperature in Kelvin and Celsius
    """
    # Input validation
    if mass_g <= 0 or current_A <= 0 or resistance_ohm <= 0 or time_s <= 0:
        raise ValueError("All input parameters must be positive numbers.")

    # Constants
    specific_heat_capacity = 4186  # J/kg.K for water

    # Convert mass from grams to kilograms
    mass_kg = mass_g / 1000.0

    # Calculate electrical energy supplied (Q = I^2 * R * t)
    energy_J = (current_A ** 2) * resistance_ohm * time_s

    # Calculate temperature change (delta T = Q / (m * c))
    delta_T = energy_J / (mass_kg * specific_heat_capacity)

    # Calculate final temperature in Kelvin
    final_temp_K = initial_temp_K + delta_T

    # Convert final temperature to Celsius
    final_temp_C = final_temp_K - 273.15

    return final_temp_K, final_temp_C


if __name__ == "__main__":
    # Given values
    mass = 82.5  # grams
    initial_temperature = 300.0  # Kelvin
    current = 1.75  # Amperes
    resistance = 25.0  # Ohms
    time = 100  # seconds

    try:
        final_temp_K, final_temp_C = calculate_final_temperature(mass, initial_temperature, current, resistance, time)

        # Save results to a text file
        with open('final_temperature_results.txt', 'w') as f:
            f.write(f'Final Temperature: {final_temp_K:.2f} K\n')
            f.write(f'Final Temperature: {final_temp_C:.2f} C\n')
    except Exception as e:
        print(f'Error: {e}')
