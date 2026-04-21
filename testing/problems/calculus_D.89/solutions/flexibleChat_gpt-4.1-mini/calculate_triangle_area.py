# filename: calculate_triangle_area.py
import math

def calculate_triangle_area(side1, side2, angle_degrees):
    """Calculate the area of a triangle given two sides and the included angle in degrees."""
    # Validate inputs
    if side1 <= 0 or side2 <= 0:
        raise ValueError("Side lengths must be positive numbers.")
    if not (0 < angle_degrees < 180):
        raise ValueError("Angle must be between 0 and 180 degrees (exclusive).")

    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)

    # Calculate area using formula: (1/2) * side1 * side2 * sin(angle)
    area = 0.5 * side1 * side2 * math.sin(angle_radians)

    # Round the result to five decimal places
    return round(area, 5)

# Given values
AB = 10  # cm
BC = 3   # cm
angle_ABC = 107  # degrees

# Calculate and print the area
area_ABC = calculate_triangle_area(AB, BC, angle_ABC)
print(area_ABC)