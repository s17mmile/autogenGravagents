import math

def calculate_damped_frequency_ratio(cycles):
    # Calculate the damping ratio
    damping_ratio = 1 / (8 * math.pi)
    # Check for valid damping ratio
    if damping_ratio >= 1:
        raise ValueError('Damping ratio must be less than 1 for real frequency.')
    # Calculate the ratio of damped frequency to natural frequency
    return math.sqrt(1 - damping_ratio**2)

# Number of cycles
cycles = 4
# Call the function and output the result
ratio = calculate_damped_frequency_ratio(cycles)
print(f'The ratio of the frequency of the damped oscillator to its natural frequency is approximately: {ratio:.6f}')