# filename: calculate_muscle_work.py

def calculate_work(k, x_cm):
    """Calculate the work done by a spring-like muscle fiber.

    Args:
        k (float): Spring constant in N/m, must be non-negative.
        x_cm (float): Displacement in centimeters, must be non-negative.

    Returns:
        float: Work done in joules.
    """
    if k < 0 or x_cm < 0:
        raise ValueError("Spring constant and displacement must be non-negative.")
    # Convert displacement from cm to m
    x_m = x_cm / 100.0
    # Calculate work done using W = 0.5 * k * x^2
    work = 0.5 * k * x_m ** 2
    return work

# Given values
k = 750  # N/m
x_cm = 3.5  # cm

work_done = calculate_work(k, x_cm)
print(f"Work performed by the muscle fiber: {work_done:.4f} Joules")
