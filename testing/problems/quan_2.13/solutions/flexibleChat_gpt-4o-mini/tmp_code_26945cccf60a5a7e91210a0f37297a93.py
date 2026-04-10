import math

# Constants
h = 6.626e-34  # Planck's constant in Js
m = 9.1e-31    # Mass of the particle in kg
f = 6.0e14     # Frequency of the emitted photon in s^-1

def calculate_box_length(h, m, f):
    try:
        # Calculate the length of the box
        L_squared = (21 * h) / (8 * m * f)
        if L_squared < 0:
            raise ValueError('Calculated length squared is negative, check input values.')
        L = math.sqrt(L_squared)
        # Convert length to nanometers
        L_nm = L * 1e9
        return L_nm
    except ZeroDivisionError:
        return 'Error: Division by zero encountered in calculations.'
    except Exception as e:
        return f'Error: {str(e)}'

# Calculate and print the length of the box
length_of_box = calculate_box_length(h, m, f)
print(f'Length of the box: {length_of_box:.2f} nm')