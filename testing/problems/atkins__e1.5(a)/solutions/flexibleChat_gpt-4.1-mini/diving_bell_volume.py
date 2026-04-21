# filename: diving_bell_volume.py

def calculate_diving_bell_volume(V1=3.0, depth=50, density_seawater=1.025, g=9.81, P1=101325):
    """
    Calculate the volume of air space in a diving bell at a given depth assuming constant temperature.

    Parameters:
    - V1: Initial volume at surface in cubic meters
    - depth: Depth in meters
    - density_seawater: Density of seawater in g/cm^3
    - g: Acceleration due to gravity in m/s^2
    - P1: Atmospheric pressure at surface in Pascals

    Returns:
    - V2: Volume at given depth in cubic meters

    Assumes isothermal conditions (constant temperature).
    """
    # Basic input validation
    if V1 <= 0 or depth < 0 or density_seawater <= 0 or g <= 0 or P1 <= 0:
        raise ValueError("All input parameters must be positive numbers, and depth cannot be negative.")

    # Convert density from g/cm^3 to kg/m^3
    density_kg_m3 = density_seawater * 1000

    # Calculate pressure due to water column
    P_water = density_kg_m3 * g * depth

    # Total pressure at depth
    P2 = P1 + P_water

    # Apply Boyle's Law: P1 * V1 = P2 * V2
    V2 = (P1 * V1) / P2

    return V2

# Calculate and print the volume at 50 m depth
volume_at_depth = calculate_diving_bell_volume()
print(f"Volume of air space at 50 m depth: {volume_at_depth:.3f} cubic meters")
