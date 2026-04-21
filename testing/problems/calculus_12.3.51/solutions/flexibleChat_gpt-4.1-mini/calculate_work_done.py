# filename: calculate_work_done.py
import math

def calculate_work(force_lb, displacement_ft, angle_deg):
    """Calculate the work done by a force acting at an angle.

    Args:
        force_lb (float): Magnitude of the force in pounds (must be >= 0).
        displacement_ft (float): Displacement in feet (must be >= 0).
        angle_deg (float): Angle of the force above horizontal in degrees (0 <= angle <= 180).

    Returns:
        float: Work done in foot-pounds.
    """
    if force_lb < 0 or displacement_ft < 0:
        raise ValueError("Force and displacement must be non-negative.")
    if not (0 <= angle_deg <= 180):
        raise ValueError("Angle must be between 0 and 180 degrees.")

    angle_rad = math.radians(angle_deg)
    work = force_lb * displacement_ft * math.cos(angle_rad)
    return work

if __name__ == "__main__":
    # Given values
    force_lb = 30
    angle_deg = 40
    displacement_ft = 80

    work_done_ft_lb = calculate_work(force_lb, displacement_ft, angle_deg)
    print(f"Work done by the force: {work_done_ft_lb:.2f} foot-pounds")
