def estimate_liver_volume(areas, interval_length):
    """
    Estimate the volume of the liver using the Midpoint Rule.
    The areas correspond to cross-sectional areas of the liver at specified intervals.
    """
    # Calculate the number of intervals
    n_intervals = len(areas) - 1
    volume = 0

    # Calculate the volume using the Midpoint Rule
    for i in range(n_intervals):
        # Area at the right endpoint of the interval
        right_endpoint_area = areas[i + 1]  # Area at the right endpoint
        volume += right_endpoint_area * interval_length

    return volume

# Constants
interval_length = 1.5  # cm
areas = [0, 18, 58, 79, 94, 106, 117, 128, 63, 39, 0]  # Areas in cm²

# Estimate the volume of the liver
estimated_volume = estimate_liver_volume(areas, interval_length)

# Output the estimated volume
print(f'Estimated volume of the liver: {estimated_volume} cm³')