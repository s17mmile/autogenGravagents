import math

def calculate_B1_strength(pulse_duration, flip_angle_degrees):
    # Check for valid input
    if pulse_duration <= 0:
        raise ValueError('Pulse duration must be greater than zero.')
    if flip_angle_degrees <= 0:
        raise ValueError('Flip angle must be greater than zero.')

    # Convert flip angle to radians
    flip_angle_radians = math.radians(flip_angle_degrees)

    # Gyromagnetic ratio for protons (in rad/s/T)
    gamma = 2.675e8  # gyromagnetic ratio for protons

    # Calculate B1 field strength
    B1_strength = flip_angle_radians / (gamma * pulse_duration)

    # Convert to milliTesla
    B1_strength_mT = B1_strength * 1000  # convert from Tesla to milliTesla
    return B1_strength_mT

# Example usage
pulse_duration = 12.5e-6  # pulse duration in seconds (12.5 microseconds)
flip_angle_degrees = 180  # flip angle in degrees

# Calculate and output the B1 field strength
B1_strength_mT = calculate_B1_strength(pulse_duration, flip_angle_degrees)
print(f'Strength of B1 field: {B1_strength_mT:.5f} mT')