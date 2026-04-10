def calculate_entropy_change(heat_transferred, temperature):
    if temperature <= 0:
        raise ValueError('Temperature must be greater than 0 K')
    return heat_transferred / temperature

# Constants
heat_transferred = 25000  # Heat transferred in Joules (25 kJ)
temperature = 273.15       # Temperature in Kelvin (0 degrees Celsius)

# Calculate change in entropy
Delta_S = calculate_entropy_change(heat_transferred, temperature)

# Print the result
print(f'Change in entropy: {Delta_S:.2f} J/K')