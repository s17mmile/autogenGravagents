# filename: calculate_building_height.py

def calculate_building_height(total_time=4.021, speed_of_sound=331, g=9.81):
    # Validate inputs
    if total_time <= 0 or speed_of_sound <= 0 or g <= 0:
        raise ValueError("All input parameters must be positive numbers.")

    # Define the function for which we want to find the root: f(t_fall) = t_fall + (0.5*g*t_fall**2)/speed_of_sound - total_time
    def f(t):
        return t + (0.5 * g * t**2) / speed_of_sound - total_time

    # Derivative of f(t) with respect to t
    def df(t):
        return 1 + (g * t) / speed_of_sound

    # Initial guess for t_fall
    t_fall = total_time / 2

    max_iterations = 20
    tolerance = 1e-10

    for _ in range(max_iterations):
        t_fall_new = t_fall - f(t_fall) / df(t_fall)
        if abs(t_fall_new - t_fall) < tolerance:
            t_fall = t_fall_new
            break
        t_fall = t_fall_new
    else:
        print("Warning: Newton-Raphson method did not converge within the maximum iterations.")

    # Calculate height
    h = 0.5 * g * t_fall**2
    return t_fall, h

# Calculate and print the height
fall_time, height = calculate_building_height()
print(f"Fall time: {fall_time:.6f} seconds")
print(f"Height of the building: {height:.3f} meters")
