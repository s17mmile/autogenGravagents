import numpy as np

def calculate_string_length(radius, thickness, gap_fraction):
    # Calculate the diameter of the sphere
    diameter = 2 * radius  
    # Effective thickness considering gaps
    effective_thickness = thickness * (1 + gap_fraction)
    
    # Check if effective thickness exceeds diameter
    if effective_thickness > diameter:
        raise ValueError('Effective thickness exceeds the diameter of the sphere.')  
    
    # Calculate the number of layers
    number_of_layers = int(diameter / effective_thickness)
    
    # Calculate total length of the string
    total_length = 0.0
    for layer in range(number_of_layers):
        # Calculate the radius at this layer
        current_radius = radius + layer * effective_thickness
        # Calculate the circumference at this layer
        circumference = 2 * np.pi * current_radius
        # Add to total length
        total_length += circumference
    
    return total_length

# Constants
radius = 2.0  # radius of the sphere in meters
thickness = 0.0075  # thickness of the string in meters
gap_fraction = 0.1  # 10% gap between layers

# Calculate and output the total length of the string
try:
    total_length = calculate_string_length(radius, thickness, gap_fraction)
    print(f'Total length of the string: {total_length:.2f} meters')
except ValueError as e:
    print(e)