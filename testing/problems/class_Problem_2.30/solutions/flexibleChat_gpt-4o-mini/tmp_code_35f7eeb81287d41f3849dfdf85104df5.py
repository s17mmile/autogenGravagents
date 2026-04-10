import math

def calculate_building_height(total_time, g=9.81, speed_of_sound=331):
    # Coefficients for the quadratic equation
    a = 4.905
    b = 331
    c = -1337.951  # Corrected constant term

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        print("Error: The discriminant is negative, no real roots exist.")
        return None

    # Calculate the positive root using the quadratic formula
    t_f = (-b + math.sqrt(discriminant)) / (2 * a)

    # Calculate the height using the free fall equation
    h = 0.5 * g * t_f**2
    return h

# Total time in seconds
total_time = 4.021
height = calculate_building_height(total_time)
if height is not None:
    print(f'The height of the building is approximately {height:.2f} meters.')