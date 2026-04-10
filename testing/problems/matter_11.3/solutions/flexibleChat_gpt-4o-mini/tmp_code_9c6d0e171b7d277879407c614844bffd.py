import numpy as np

# Constants
h = 6.626e-34  # Planck's constant (Js)
m = 9.11e-31   # Mass of an electron (kg)
c = 3.00e8     # Speed of light (m/s)
L = 0.7e-9     # Length of the sides of the cube (m)

def calculate_wavelength():
    """
    Calculate the predicted wavelength of the first excitation of the buckminsterfullerene molecule (C60)
    using the quantum mechanics model of a particle in a three-dimensional box.
    """
    # Energy difference calculation
    Delta_E = (5 * h**2) / (8 * m * L**2)
    
    # Wavelength calculation
    wavelength = (8 * m * L**2 * c) / (5 * h)
    
    # Output the predicted wavelength
    predicted_wavelength_nm = wavelength * 1e9  # Convert to nanometers
    print(f'Predicted wavelength for the first excitation of C60: {predicted_wavelength_nm:.2f} nm')
    print(f'Actual wavelength: 730 nm')
    print(f'Prediction accuracy: {predicted_wavelength_nm / 730 * 100:.2f}% of the actual value')

# Execute the function
calculate_wavelength()