# filename: liver_volume_midpoint_function.py

def estimate_liver_volume_midpoint(areas, spacing):
    """Estimate the volume of the liver using the Midpoint Rule.

    Args:
        areas (list of float): Cross-sectional areas in square centimeters.
        spacing (float): Distance between cross-sections in centimeters.

    Returns:
        float: Estimated volume in cubic centimeters.
    """
    if not areas or spacing <= 0:
        raise ValueError("Areas list must not be empty and spacing must be positive.")
    
    midpoint_areas = [(areas[i] + areas[i+1]) / 2 for i in range(len(areas) - 1)]
    volume = sum(midpoint_areas) * spacing
    return volume


# Given data
spacing = 1.5  # cm
areas = [0, 18, 58, 79, 94, 106, 117, 128, 63, 39, 0]  # cross-sectional areas in cm^2

# Calculate volume estimate
volume_estimate = estimate_liver_volume_midpoint(areas, spacing)

# Output the result
print(f"Estimated volume of the liver using Midpoint Rule: {volume_estimate:.2f} cubic centimeters")
