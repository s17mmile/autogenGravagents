# filename: max_drop_height_with_drag.py
import math

def max_drop_height(mass, drag_coeff, max_velocity, g=9.8):
    """Calculate the maximum height from which an object can be dropped so that its velocity does not exceed max_velocity.

    Args:
        mass (float): Mass of the object in kg.
        drag_coeff (float): Drag coefficient (N per m/s).
        max_velocity (float): Maximum allowed velocity in m/s.
        g (float, optional): Acceleration due to gravity in m/s^2. Defaults to 9.8.

    Returns:
        float: Maximum height in meters.

    Raises:
        ValueError: If max_velocity exceeds terminal velocity.
    """
    # Calculate terminal velocity
    v_terminal = mass * g / drag_coeff

    # Check if max_velocity is less than terminal velocity
    if max_velocity > v_terminal:
        raise ValueError("Maximum velocity exceeds terminal velocity; no maximum height limit.")

    # Calculate time to reach max_velocity
    time_to_max_v = (mass / drag_coeff) * (-math.log(1 - max_velocity / v_terminal))

    # Calculate height fallen by integrating velocity over time
    height = v_terminal * (time_to_max_v + (mass / drag_coeff) * (math.exp(- (drag_coeff / mass) * time_to_max_v) - 1))

    return height

# Given parameters
mass = 0.25  # kg
drag_coeff = 0.2  # N/(m/s)
max_velocity = 10  # m/s

def main():
    height = max_drop_height(mass, drag_coeff, max_velocity)
    print(f"Maximum height to drop from without exceeding {max_velocity} m/s: {height:.2f} meters")

if __name__ == "__main__":
    main()
