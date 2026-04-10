import numpy as np
from scipy.integrate import quad

def area_of_cross_section(x):
    """
    Calculate the area of the isosceles right triangle cross-section at position x.
    """
    return 9 - x**2

def volume_of_solid():
    """
    Calculate the volume of the solid by integrating the area of the cross-sections.
    """
    try:
        volume, _ = quad(area_of_cross_section, -3, 3)
        return volume
    except Exception as e:
        print(f'An error occurred during integration: {e}') 
        return None

# Calculate the volume
solid_volume = volume_of_solid()
if solid_volume is not None:
    print(f'The volume of the solid is: {solid_volume} cubic units')