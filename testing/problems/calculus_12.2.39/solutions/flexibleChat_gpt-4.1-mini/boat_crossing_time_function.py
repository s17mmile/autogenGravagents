# filename: boat_crossing_time_function.py
import math

def calculate_boat_crossing_time(width_canal, upstream_distance, current_speed, boat_speed):
    """
    Calculate the time taken for a boat to cross a canal and land upstream,
    considering current speed and boat speed.

    Parameters:
    - width_canal: width of the canal in km
    - upstream_distance: distance upstream to land in km
    - current_speed: speed of the current downstream in km/h
    - boat_speed: speed of the boat in still water in km/h

    Returns:
    - time in minutes
    """
    c = current_speed / boat_speed
    b = 26 / 39  # approx 0.6667
    a = 1.0

    R = math.sqrt(a**2 + b**2)
    alpha = math.asin(b / R)  # radians

    sin_theta_alpha = c / R
    if abs(sin_theta_alpha) > 1:
        raise ValueError("No valid solution for the angle theta.")

    theta_alpha_1 = math.asin(sin_theta_alpha)
    theta_alpha_2 = math.pi - theta_alpha_1

    theta_1 = theta_alpha_1 - alpha
    theta_2 = theta_alpha_2 - alpha

    v_x_1 = boat_speed * math.sin(theta_1) - current_speed
    v_x_2 = boat_speed * math.sin(theta_2) - current_speed

    # Select theta with negative v_x (upstream)
    if v_x_1 < 0:
        theta = theta_1
        v_x = v_x_1
    elif v_x_2 < 0:
        theta = theta_2
        v_x = v_x_2
    else:
        raise ValueError("No valid boat heading angle found to go upstream.")

    v_y = boat_speed * math.cos(theta)
    time_hours = width_canal / v_y

    # Verify upstream displacement
    upstream_displacement = -v_x * time_hours
    if abs(upstream_displacement - upstream_distance) > 0.01:
        raise ValueError("Calculated upstream displacement does not match target.")

    print(f"Boat heading angle (degrees) relative to across-canal axis: {math.degrees(theta):.2f}")
    print(f"Time to cross the canal: {time_hours * 60:.2f} minutes")
    print(f"Upstream displacement achieved: {upstream_displacement:.2f} km (target: {upstream_distance} km)")

    return time_hours * 60

# Given data
width_canal = 3.0  # km
upstream_distance = 2.0  # km
current_speed = 3.5  # km/h
boat_speed = 13.0  # km/h

# Calculate and print the crossing time
crossing_time_minutes = calculate_boat_crossing_time(width_canal, upstream_distance, current_speed, boat_speed)
