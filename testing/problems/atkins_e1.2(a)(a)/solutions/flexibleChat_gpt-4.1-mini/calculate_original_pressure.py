# filename: calculate_original_pressure.py

def calculate_original_pressure(volume_reduction, final_volume, final_pressure):
    """Calculate the original pressure of a gas undergoing isothermal compression.

    Args:
        volume_reduction (float): Volume reduction in dm^3.
        final_volume (float): Final volume in dm^3.
        final_pressure (float): Final pressure in bar.

    Returns:
        float: Original pressure in bar.
    """
    # Input validation
    if volume_reduction < 0 or final_volume <= 0 or final_pressure <= 0:
        raise ValueError("Volumes and pressure must be positive numbers.")

    # Calculate original volume
    original_volume = final_volume + volume_reduction

    # Calculate original pressure using Boyle's Law: P1 * V1 = P2 * V2
    original_pressure = (final_pressure * final_volume) / original_volume

    return original_pressure


# Given data
volume_reduction = 2.20  # in dm^3
final_volume = 4.65       # in dm^3
final_pressure = 5.04    # in bar

# Calculate and print the original pressure
original_pressure = calculate_original_pressure(volume_reduction, final_volume, final_pressure)
print(f"Original pressure of the gas: {original_pressure:.2f} bar")
