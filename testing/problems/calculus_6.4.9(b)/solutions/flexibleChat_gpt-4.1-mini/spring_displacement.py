# filename: spring_displacement.py

def calculate_spring_displacement(work_done, initial_length_cm, stretched_length_cm, force):
    """
    Calculate how far beyond the natural length a force will stretch a spring.

    Parameters:
    work_done (float): Work done to stretch the spring in Joules.
    initial_length_cm (float): Natural length of the spring in centimeters.
    stretched_length_cm (float): Length of the spring after work is done in centimeters.
    force (float): Force applied to the spring in Newtons.

    Returns:
    tuple: spring constant (N/m), displacement caused by force in meters, displacement in centimeters
    """
    # Convert lengths to meters
    initial_length_m = initial_length_cm / 100
    stretched_length_m = stretched_length_cm / 100

    # Calculate initial displacement in meters
    displacement_initial = stretched_length_m - initial_length_m

    # Calculate spring constant k using work done formula: W = 0.5 * k * x^2
    k = (2 * work_done) / (displacement_initial ** 2)

    # Calculate displacement x caused by force F using Hooke's law: F = k * x
    displacement_force_m = force / k

    # Convert displacement to centimeters
    displacement_force_cm = displacement_force_m * 100

    return k, displacement_force_m, displacement_force_cm


# Given values
work_done = 2.0  # Joules
initial_length_cm = 30.0  # cm
stretched_length_cm = 42.0  # cm
force = 30.0  # Newtons

# Calculate results
k, displacement_m, displacement_cm = calculate_spring_displacement(work_done, initial_length_cm, stretched_length_cm, force)

# Output results with explanation
print(f"Spring constant k: {k:.2f} N/m")
print(f"A force of {force} N will stretch the spring {displacement_m:.4f} meters ({displacement_cm:.2f} cm) beyond its natural length.")
