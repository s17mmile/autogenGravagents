import numpy as np
from scipy.integrate import quad

# Define a function to calculate the width of the flat metal sheet needed for a given panel width

def calculate_width(panel_width):
    # Define the sine function based on the given equation
    def sine_wave(x):
        return np.sin(np.pi * x / 7)

    try:
        # Calculate the integral from 0 to 14
        area, _ = quad(sine_wave, 0, 14)
        # The width of the flat metal sheet needed for the specified panel width
        w = 2 * area  # Corrected to multiply by 2 for the total width
        return w
    except Exception as e:
        print(f'An error occurred during integration: {e}')
        return None

# Set the desired panel width
panel_width = 28
# Calculate the width of the flat metal sheet needed
width = calculate_width(panel_width)

# Print the result rounded to four significant digits if width is calculated
if width is not None:
    print(f'Width of the flat metal sheet needed: {width:.4f} inches')