# filename: calculate_laser_frequency.py

# Define constant for speed of light
SPEED_OF_LIGHT = 3.00e8  # meters per second

def calculate_frequency(wavelength_nm):
    """Calculate frequency of light given wavelength in nanometers."""
    wavelength_m = wavelength_nm * 1e-9  # Convert nm to meters
    frequency_hz = SPEED_OF_LIGHT / wavelength_m
    return frequency_hz

# Given wavelength
wavelength_nm = 632.8

# Calculate frequency
frequency = calculate_frequency(wavelength_nm)

# Print the frequency
print(f"Frequency of helium-neon laser light: {frequency:.2e} Hz")

# Return frequency for further use if needed
frequency
