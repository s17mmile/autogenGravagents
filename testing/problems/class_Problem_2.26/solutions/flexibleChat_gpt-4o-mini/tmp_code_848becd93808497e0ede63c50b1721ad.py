def calculate_max_spring_compression(mass, initial_speed, spring_constant):
    # Calculate initial kinetic energy (KE)
    kinetic_energy = 0.5 * mass * (initial_speed ** 2)

    # Check for valid spring constant
    if spring_constant <= 0:
        raise ValueError('Spring constant must be greater than zero.')

    # Calculate maximum compression of the spring (x)
    max_compression_squared_value = (2 * kinetic_energy) / spring_constant
    max_compression = (max_compression_squared_value ** 0.5)

    return max_compression

# Constants
mass = 2  # kg
initial_speed = 4  # m/s
spring_constant = 6  # N/m

# Calculate maximum compression
max_compression = calculate_max_spring_compression(mass, initial_speed, spring_constant)

# Print the result
print(f'Maximum compression of the spring: {max_compression:.2f} m')