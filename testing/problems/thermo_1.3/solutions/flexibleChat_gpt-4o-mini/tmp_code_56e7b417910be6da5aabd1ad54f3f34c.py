def calculate_argon_pressure(n, Vm, T, a, b):
    # Constants
    R = 0.08314  # L bar K^-1 mol^-1

    # Calculate pressure using van der Waals equation
    try:
        repulsive_pressure = (n * R * T) / (Vm - n * b)  # Repulsive term
        attractive_pressure = (a * n**2) / (Vm**2)  # Attractive term
        total_pressure = repulsive_pressure - attractive_pressure  # Total pressure
    except ZeroDivisionError:
        return 'Error: Division by zero encountered in pressure calculation.'

    # Determine dominant potential
    dominant_force = 'Repulsive' if repulsive_pressure > attractive_pressure else 'Attractive'

    # Return results
    return total_pressure, dominant_force

# Parameters for Argon
n = 1  # moles of Ar
Vm = 1.31  # L mol^-1
T = 426  # K

a = 1.355  # bar dm^6 mol^-2
b = 0.0320  # L mol^-1

# Calculate and print results
pressure, force = calculate_argon_pressure(n, Vm, T, a, b)
print(f'Pressure exerted by Argon using van der Waals equation: {pressure:.3f} bar')
print(f'Dominant potential: {force}')